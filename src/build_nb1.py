"""Builds notebooks/1_data_processing.ipynb — PySpark, dual-mode (local + Databricks).

Desenho: unidade (cliente x onda); tratamento W = envio; controle limpo por onda;
target resposta = viu & usou (ordem t_view <= t_comp); outcome contínuo = gasto na
janela; features estritamente pré-onda.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# 1 · Processamento de Dados — dataset unificado (PySpark)

**Objetivo:** transformar os 3 arquivos brutos em uma **tabela de modelagem (cliente × onda)**
que sustenta tanto o modelo de resposta quanto a leitura causal do envio.

## Desenho (decisões e alternativas descartadas)

| Decisão | Justificativa | Alternativa descartada |
|---|---|---|
| **Unidade = (cliente × onda)** | Os envios ocorrem em 6 ondas (dias 0,7,14,17,21,24) e cada cliente recebe **no máx. 1 oferta por onda** (verificado adiante). A linha fica não-ambígua: tratado (com atributos da oferta) ou controle. | (cliente × oferta recebida): sem linhas de controle não se estima efeito do envio; 56% das janelas se sobrepõem, atribuição ambígua. |
| **Tratamento W = envio na onda** | É a alavanca que o negócio controla. ~25% da base não recebe nada em cada onda e o *balance check* (adiante) sugere aleatorização → controle utilizável. | W = visualizou: é **pós-tratamento** (mediador); condicionar nele embute viés de seleção. |
| **Controle limpo** = não recebeu na onda **e** sem janela anterior ativa | Cliente sem oferta vigente é contrafactual do "não enviar". | Usar todos os não-recebedores: contaminados por ofertas anteriores ainda ativas. |
| **Target resposta `y_response` = viu E usou** (`t_view ≤ t_comp`, na validade) | "Usar" = `offer completed` (resgate). Sem exigir visualização **anterior**, 16,3% das instâncias seriam falsos sucessos (auto-resgate: 9,4% sem ver + 6,9% viu depois). Informational: viu E transacionou após ver. | `completed` puro: contamina o target com compras que ocorreriam de qualquer forma. |
| **Outcome contínuo = gasto em horizontes fixos** (3/4/5/7/10d) | Permite contraste tratado × controle com **janelas do mesmo tamanho** (controle não tem `duration`). | Gasto na janela da oferta apenas: sem equivalente no controle. |
| **Features estritamente pré-onda** (`t < dia da onda`) | Fecha vazamento temporal. Onda 0 fica sem histórico → flag `has_history`. | Agregados do período todo: vazam o próprio desfecho. |
| **Split temporal** (treino ondas 0–14, teste 17–24; aplicado no NB2) | Simula decisão real: treinar no passado, decidir no futuro. | Split aleatório: vaza o futuro e superestima performance. |

## Premissas
1. `age == 118` é sentinela de perfil incompleto (coincide 100% com nulos de `gender`/`credit_card_limit`).
2. Janela de validade = `[t_recv, t_recv + duration]` dias; `time_since_test_start` em dias (0–29.75).
3. `offer completed` = cupom usado (resgate ao atingir `min_value` na validade).
4. Envio nas ondas ≈ aleatorizado (suportado pelo balance check; validação definitiva = A/B).
""")

md("""## Setup (dual-mode: local ou Databricks)
O notebook roda **sem alterações** em ambiente local ou no Databricks (Free Edition/
Community): detecta o ambiente, obtém a sessão Spark via `getOrCreate` e, se os
arquivos brutos não existirem no disco, **baixa o tar.gz da URL pública do S3**.""")
co("""import os, sys, glob, tarfile, tempfile, urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import types as T

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    spark = SparkSession.builder.getOrCreate()   # sessão provida pela plataforma
else:
    # Windows/local: aponta os workers Python do Spark para o interpretador atual
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    # heap do driver precisa ser definido ANTES da JVM subir (config no builder é ignorada)
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 4g pyspark-shell")
    spark = (SparkSession.builder.master("local[*]").appName("ifood-nb1")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.enabled", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

CWD = Path(os.getcwd())
ROOT = CWD.parent if CWD.name == "notebooks" else CWD
DATA_URL = "https://data-architect-test-source.s3.sa-east-1.amazonaws.com/ds-technical-evaluation-data.tar.gz"
print("Ambiente:", "Databricks" if IS_DATABRICKS else "local", "| Spark", spark.version)""")

co("""def ensure_raw_data():
    \"\"\"Retorna dir com offers/profile/transactions.json; baixa do S3 se preciso.\"\"\"
    raw = ROOT / "data" / "raw"
    if (raw / "transactions.json").exists():
        return raw
    tmp = Path(tempfile.gettempdir()) / "ifood_raw"
    if not (tmp / "transactions.json").exists():
        tmp.mkdir(parents=True, exist_ok=True)
        tgz = tmp / "data.tar.gz"
        print("Baixando dados de", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(tmp)
        for f in glob.glob(str(tmp / "**" / "*.json"), recursive=True):
            Path(f).replace(tmp / Path(f).name)
    return tmp

RAW = ensure_raw_data()
print("Dados em:", RAW)""")

md("""## Ingestão
Parse do campo aninhado `value` (`offer id` com espaço em received/viewed,
`offer_id` em completed, `amount` em transações) feito em pandas — caminho de
ingestão único que funciona igual local e no serverless do Databricks — e conversão
para Spark **com schema explícito**. Todo o processamento pesado é PySpark.""")
co("""offers_pd = pd.read_json(RAW / "offers.json")
profile_pd = pd.read_json(RAW / "profile.json")
tx_pd = pd.read_json(RAW / "transactions.json")

tx_pd["offer_id"] = tx_pd["value"].apply(lambda d: d.get("offer id") or d.get("offer_id"))
tx_pd["amount"] = pd.to_numeric(tx_pd["value"].apply(lambda d: d.get("amount")))
tx_pd["reward"] = pd.to_numeric(tx_pd["value"].apply(lambda d: d.get("reward")))
tx_flat = tx_pd[["account_id", "event", "offer_id", "amount", "reward",
                 "time_since_test_start"]].rename(columns={"time_since_test_start": "t"})
offers_pd["channels_str"] = offers_pd["channels"].apply(",".join)
profile_pd["registered_on"] = profile_pd["registered_on"].astype(str)

def none_ify(df):
    return df.astype(object).where(df.notna(), None)

tx_s = spark.createDataFrame(none_ify(tx_flat), schema=T.StructType([
    T.StructField("account_id", T.StringType()), T.StructField("event", T.StringType()),
    T.StructField("offer_id", T.StringType()), T.StructField("amount", T.DoubleType()),
    T.StructField("reward", T.DoubleType()), T.StructField("t", T.DoubleType())]))
offers_s = spark.createDataFrame(
    none_ify(offers_pd[["id", "offer_type", "min_value", "discount_value", "duration", "channels_str"]]),
    schema=T.StructType([
        T.StructField("offer_id", T.StringType()), T.StructField("offer_type", T.StringType()),
        T.StructField("min_value", T.DoubleType()), T.StructField("discount_value", T.DoubleType()),
        T.StructField("duration", T.DoubleType()), T.StructField("channels_str", T.StringType())]))
profile_s = spark.createDataFrame(
    none_ify(profile_pd[["id", "age", "gender", "credit_card_limit", "registered_on"]]),
    schema=T.StructType([
        T.StructField("account_id", T.StringType()), T.StructField("age", T.DoubleType()),
        T.StructField("gender", T.StringType()), T.StructField("credit_card_limit", T.DoubleType()),
        T.StructField("registered_on", T.StringType())]))
# cache: os DFs vêm de coleções locais (LocalRelation); sem cache, cada join
# re-embute e recomputa os dados no plano — custo de memória desnecessário.
tx_s = tx_s.repartition(8).cache()
offers_s = offers_s.cache()
profile_s = profile_s.cache()
print("tx:", tx_s.count(), "| offers:", offers_s.count(), "| profile:", profile_s.count())""")

md("## Limpeza — perfil e ofertas")
co("""CHANNELS = ["web", "email", "mobile", "social"]
offers_c = offers_s.withColumn("channels", F.split("channels_str", ","))
for ch in CHANNELS:
    offers_c = offers_c.withColumn(f"ch_{ch}", F.array_contains("channels", ch).cast("int"))
offers_c = offers_c.withColumn("n_channels", sum(F.col(f"ch_{c}") for c in CHANNELS)) \\
                   .drop("channels", "channels_str")

ref_date = profile_s.select(F.max(F.to_date("registered_on", "yyyyMMdd"))).first()[0]
profile_c = (profile_s
    .withColumn("incomplete_profile", (F.col("age") == 118).cast("int"))
    .withColumn("age", F.when(F.col("age") == 118, None).otherwise(F.col("age")))
    .withColumn("gender", F.coalesce("gender", F.lit("unknown")))
    .withColumn("account_age_days",
                F.datediff(F.lit(ref_date), F.to_date("registered_on", "yyyyMMdd")))
    .drop("registered_on"))
profile_c.show(3)""")

md("""## Ondas de envio e verificação da unidade
Confirmamos empiricamente: 6 ondas e **no máximo 1 oferta por cliente por onda** —
o que valida a unidade (cliente × onda).""")
co("""events = {e: tx_s.filter(F.col("event") == e).drop("event")
          for e in ["offer received", "offer viewed", "offer completed", "transaction"]}
rec = events["offer received"].select("account_id", "offer_id", F.col("t").alias("t_recv"))
vie = events["offer viewed"].select("account_id", "offer_id", F.col("t").alias("t_view"))
com = events["offer completed"].select("account_id", "offer_id", F.col("t").alias("t_comp"))
trans = events["transaction"].select("account_id", F.col("t").alias("t_tx"), "amount")

waves = [r[0] for r in rec.select("t_recv").distinct().orderBy("t_recv").collect()]
print("ondas:", waves)
dup = (rec.groupBy("account_id", "t_recv").count().filter("count > 1").count())
assert len(waves) == 6 and dup == 0, "premissa violada"
print("máx. 1 oferta por (cliente, onda): OK")""")

md("""## Grade (cliente × onda) + tratamento
17.000 clientes × 6 ondas = 102.000 linhas. `W=1` se recebeu oferta na onda.
`n_active_offers` = janelas de ondas anteriores ainda vigentes no dia do envio —
define o **controle limpo** (`W=0` e nenhuma oferta ativa).""")
co("""waves_s = spark.createDataFrame([(float(w),) for w in waves], ["wave"])
grid = profile_c.select("account_id").crossJoin(waves_s)

inst = (rec.join(offers_c, "offer_id")
           .withColumn("t_end", F.col("t_recv") + F.col("duration")))

active = (grid.join(inst.select("account_id", F.col("t_recv").alias("t0"), "t_end"), "account_id")
              .filter((F.col("t0") < F.col("wave")) & (F.col("t_end") > F.col("wave")))
              .groupBy("account_id", "wave").agg(F.count("*").alias("n_active_offers")))

grid = (grid.join(inst.withColumnRenamed("t_recv", "wave"), ["account_id", "wave"], "left")
            .join(active, ["account_id", "wave"], "left")
            .fillna({"n_active_offers": 0})
            .withColumn("W", F.col("offer_id").isNotNull().cast("int"))
            .withColumn("control_clean",
                        ((F.col("W") == 0) & (F.col("n_active_offers") == 0)).cast("int"))
            .cache())
print("grade:", grid.count())
grid.groupBy("wave").agg(F.sum("W").alias("tratados"),
                         F.sum(F.when(F.col("W") == 0, 1).otherwise(0)).alias("nao_recebeu"),
                         F.sum("control_clean").alias("controle_limpo")).orderBy("wave").show()""")

md("""## Atribuição na janela da oferta (tratados)
`first_view`/`first_comp` do **mesmo offer_id** dentro de `[t_recv, t_end]`.
O flag `view_before_comp` implementa a ordem exigida pelo target.""")
co("""base_t = grid.filter("W = 1").select("account_id", "wave", "offer_id", "t_end", "offer_type",
                                     "discount_value")

fv = (base_t.join(vie, ["account_id", "offer_id"])
            .filter((F.col("t_view") >= F.col("wave")) & (F.col("t_view") <= F.col("t_end")))
            .groupBy("account_id", "wave").agg(F.min("t_view").alias("first_view")))
fc = (base_t.join(com, ["account_id", "offer_id"])
            .filter((F.col("t_comp") >= F.col("wave")) & (F.col("t_comp") <= F.col("t_end")))
            .groupBy("account_id", "wave").agg(F.min("t_comp").alias("first_comp")))

attrib = (base_t.join(fv, ["account_id", "wave"], "left")
                .join(fc, ["account_id", "wave"], "left")
                .withColumn("viewed", F.col("first_view").isNotNull().cast("int"))
                .withColumn("completed", F.col("first_comp").isNotNull().cast("int"))
                .withColumn("view_before_comp",
                            ((F.col("viewed") == 1) & (F.col("completed") == 1) &
                             (F.col("first_view") <= F.col("first_comp"))).cast("int"))
                .withColumn("reward_paid",
                            F.when(F.col("completed") == 1, F.col("discount_value")).otherwise(0.0)))

# informational: transação APÓS a visualização, dentro da janela
tx_after_view = (attrib.filter("offer_type = 'informational' and viewed = 1")
    .select("account_id", "wave", "first_view", "t_end")
    .join(trans, "account_id")
    .filter((F.col("t_tx") >= F.col("first_view")) & (F.col("t_tx") <= F.col("t_end")))
    .groupBy("account_id", "wave").agg(F.count("*").alias("n_tx_after_view")))

attrib = attrib.join(tx_after_view, ["account_id", "wave"], "left") \\
               .fillna({"n_tx_after_view": 0})
attrib.groupBy("offer_type").agg(
    F.mean("viewed").alias("view_rate"), F.mean("completed").alias("comp_rate"),
    F.mean("view_before_comp").alias("viu_e_usou")).show()""")

md("""## Outcomes em horizontes fixos (todas as linhas, tratado e controle)
Gasto e nº de transações em `[onda, onda+h]` para h ∈ {3,4,5,7,10} — janelas
idênticas para tratados e controles, viabilizando o contraste causal. Para tratados,
também o gasto na **janela da própria oferta** `[onda, t_end]`.""")
co("""HORIZONS = [3, 4, 5, 7, 10]
tx_out = (grid.select("account_id", "wave", "t_end")
              .join(trans, "account_id")
              .filter((F.col("t_tx") >= F.col("wave")) & (F.col("t_tx") <= F.col("wave") + 10)))
aggs = []
for h in HORIZONS:
    cond = F.col("t_tx") <= F.col("wave") + h
    aggs += [F.sum(F.when(cond, F.col("amount")).otherwise(0.0)).alias(f"spend_h{h}"),
             F.sum(F.when(cond, 1).otherwise(0)).alias(f"n_tx_h{h}")]
aggs += [F.sum(F.when(F.col("t_tx") <= F.col("t_end"), F.col("amount")).otherwise(0.0)).alias("spend_window"),
         F.sum(F.when(F.col("t_tx") <= F.col("t_end"), 1).otherwise(0)).alias("n_tx_window")]
outcomes = tx_out.groupBy("account_id", "wave").agg(*aggs)
print("linhas com alguma transação no horizonte:", outcomes.count())""")

md("""## Features estritamente pré-onda (`t < onda`)
RFM transacional + histórico de ofertas do cliente **antes** do envio. Onda 0 não
tem histórico → `has_history = 0` (mantida com flag; descartá-la custaria 1/6 dos
tratados e o controle mais limpo).""")
co("""tx_pre = (grid.select("account_id", "wave").join(trans, "account_id")
              .filter(F.col("t_tx") < F.col("wave")))
rfm = tx_pre.groupBy("account_id", "wave").agg(
    F.count("*").alias("n_tx_pre"),
    F.round(F.sum("amount"), 2).alias("spend_pre"),
    F.round(F.avg("amount"), 2).alias("avg_ticket_pre"),
    F.countDistinct(F.floor("t_tx")).alias("active_days_pre"),
    (F.first("wave") - F.max("t_tx")).alias("recency_days"))

hist_r = (grid.select("account_id", "wave").join(rec, "account_id")
              .filter(F.col("t_recv") < F.col("wave"))
              .groupBy("account_id", "wave").agg(F.count("*").alias("n_received_pre")))
hist_v = (grid.select("account_id", "wave").join(vie, "account_id")
              .filter(F.col("t_view") < F.col("wave"))
              .groupBy("account_id", "wave").agg(F.count("*").alias("n_viewed_pre")))
hist_c = (grid.select("account_id", "wave").join(com, "account_id")
              .filter(F.col("t_comp") < F.col("wave"))
              .groupBy("account_id", "wave").agg(F.count("*").alias("n_completed_pre")))
print("features pré-onda prontas")""")

md("## Montagem da tabela de modelagem")
co("""modeling = (grid
    .join(attrib.select("account_id", "wave", "first_view", "first_comp", "viewed",
                        "completed", "view_before_comp", "reward_paid", "n_tx_after_view"),
          ["account_id", "wave"], "left")
    .join(outcomes, ["account_id", "wave"], "left")
    .join(rfm, ["account_id", "wave"], "left")
    .join(hist_r, ["account_id", "wave"], "left")
    .join(hist_v, ["account_id", "wave"], "left")
    .join(hist_c, ["account_id", "wave"], "left")
    .join(profile_c, "account_id", "left")
    .fillna({"viewed": 0, "completed": 0, "view_before_comp": 0, "reward_paid": 0.0,
             "n_tx_after_view": 0, "spend_window": 0.0, "n_tx_window": 0,
             "n_tx_pre": 0, "spend_pre": 0.0, "avg_ticket_pre": 0.0, "active_days_pre": 0,
             "n_received_pre": 0, "n_viewed_pre": 0, "n_completed_pre": 0,
             **{f"spend_h{h}": 0.0 for h in HORIZONS}, **{f"n_tx_h{h}": 0 for h in HORIZONS}})
    .withColumn("has_history", (F.col("wave") > 0).cast("int"))
    .withColumn("view_rate_pre",
                F.when(F.col("n_received_pre") > 0,
                       F.col("n_viewed_pre") / F.col("n_received_pre")).otherwise(None))
    .withColumn("comp_rate_pre",
                F.when(F.col("n_received_pre") > 0,
                       F.col("n_completed_pre") / F.col("n_received_pre")).otherwise(None))
    # target resposta: viu E usou (ordem); informational: viu E transacionou após ver
    .withColumn("y_response",
                F.when(F.col("W") == 0, F.lit(0))
                 .when(F.col("offer_type") == "informational",
                       ((F.col("viewed") == 1) & (F.col("n_tx_after_view") > 0)).cast("int"))
                 .otherwise(F.col("view_before_comp")))
    # outcome líquido na janela da oferta (tratados)
    .withColumn("y_net_window",
                F.when(F.col("W") == 1, F.col("spend_window") - F.col("reward_paid"))))

n_rows = modeling.count()
print("tabela de modelagem:", n_rows, "linhas x", len(modeling.columns), "colunas")
modeling.groupBy("W").agg(F.count("*").alias("n"), F.mean("y_response").alias("y_resp"),
                          F.mean("spend_h7").alias("spend_h7")).show()""")

md("""## Sanity checks
1. Batem com a EDA estrutural (funil com ordem, controle limpo por onda)?
2. **Balance check** (onda 17): tratados vs controle limpo em features pré-onda —
   diferenças pequenas suportam a premissa de envio ~aleatório.""")
co("""bd = modeling.filter("W = 1 and offer_type in ('bogo','discount')")
tot = bd.count()
print(f"instâncias bogo/discount: {tot:,}")
for lbl, cond in [("viu e usou (ordem ok)", "view_before_comp = 1"),
                  ("completou SEM ver", "completed = 1 and viewed = 0"),
                  ("completou antes de ver", "completed = 1 and viewed = 1 and view_before_comp = 0")]:
    v = bd.filter(cond).count()
    print(f"  {lbl:24s}: {v:6,} ({v/tot:6.1%})")""")
co("""bal = (modeling.filter("wave = 17 and (W = 1 or control_clean = 1)")
    .groupBy("W").agg(F.count("*").alias("n"),
                      F.round(F.mean("spend_pre"), 2).alias("spend_pre"),
                      F.round(F.mean("n_tx_pre"), 2).alias("n_tx_pre"),
                      F.round(F.mean("age"), 1).alias("age"),
                      F.round(F.mean("incomplete_profile"), 3).alias("perfil_incompleto")))
bal.orderBy("W").show()""")

md("""## Persistência
Local: parquet em `data/processed/` (consumido pelo NB2). No Databricks a tabela
fica em `/tmp` (o NB2 roda localmente a partir do repo).""")
co("""out_dir = ROOT / "data" / "processed"
try:
    out_dir.mkdir(parents=True, exist_ok=True)
    test = out_dir / ".write_test"; test.touch(); test.unlink()
except Exception:
    out_dir = Path(tempfile.gettempdir()) / "ifood_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

modeling_pd = modeling.toPandas()
modeling_pd.to_parquet(out_dir / "modeling_table.parquet", index=False)
offers_c.toPandas().to_parquet(out_dir / "offers.parquet", index=False)
print("salvo em", out_dir)
print(modeling_pd.shape)""")

md("""## Resumo
- **Saída:** `modeling_table.parquet` — 102.000 linhas (cliente × onda): ~76,3k
  tratadas (`W=1`, com atributos da oferta e atribuição na janela) e ~25,7k de
  controle (~15,6k limpas), com outcomes em horizontes fixos e features pré-onda.
- **Targets:** `y_response` (viu E usou, com ordem) e `y_net_window` / `spend_h*`
  (impacto contínuo).
- **Próximo (NB2):** split temporal por onda, modelo de resposta por oferta,
  contraste causal tratado × controle limpo e política de envio.""")
co("""spark.stop() if not IS_DATABRICKS else None""")

nb["cells"] = c
out = Path(__file__).resolve().parents[1] / "notebooks" / "1_data_processing.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
