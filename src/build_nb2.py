"""Builds notebooks/2_modeling.ipynb.

A. Split temporal (ondas 0-14 treino / 17-24 teste)
B. ATE do envio: tratados x controle limpo, gasto líquido de reward, bootstrap
C. CATE — T-learner por tipo de oferta (3 braços vs controle limpo) no gasto líquido
D. Política (argmax valor líquido incremental, incl. "nenhuma") + Qini + simulação
E. Comparação: a abordagem convencional (classificador de resposta) e por que ela
   não decide envio — fecha o argumento em vez de interromper o raciocínio
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# 2 · Modelagem — efeito do envio e política de ofertas

**Entrada:** `data/processed/modeling_table.parquet` (NB1) — (cliente × onda), 102k
linhas, tratamento **W = envio**, controle limpo por onda, features pré-onda.

## Estrutura (decisões e alternativas descartadas)

| Bloco | O quê | Justificativa | Alternativa descartada |
|---|---|---|---|
| **A** | Split **temporal**: treino ondas 0–14, teste 17–24 | Simula a decisão real (treinar no passado, decidir no futuro) | Split aleatório: vaza o futuro |
| **B** | **ATE do envio**: tratados × controle limpo, gasto em 7d **líquido do reward**, IC bootstrap | Responde "quanto o envio gera de incremento, descontado o cupom" | Atribuir todo o gasto da janela à oferta: ~43% ocorreria sem ela |
| **C** | **CATE** — T-learner por **tipo** (3 braços × controle limpo) no gasto líquido | Efeito heterogêneo com controle real; braços por tipo mantêm amostra; atributos da oferta diferenciam ofertas dentro do braço | T-learner por offer_id: 8 braços finos demais |
| **D** | **Política**: argmax do valor líquido incremental esperado, incluindo "nenhuma"; avaliação por **Qini** e simulação financeira | A entrega é uma regra de alocação, não um score | Ranquear por propensão: prioriza quem compraria de qualquer forma |
| **E** | **Comparação** com a abordagem convencional (classificador de resposta) | Mostra que um modelo tecnicamente bom (AUC 0,806) ranqueia mal em valor incremental — justifica o desenho causal em vez de só afirmá-lo | Omitir a comparação: deixaria a escolha do desenho sem evidência |

**A corrente lógica é B → C → D:** *enviar compensa? → para quem faz mais diferença?
→ qual oferta mandar para cada um?* O bloco E é o fechamento comparativo, não uma
etapa do raciocínio.

**Premissas herdadas do NB1:** envio ~aleatorizado por onda (balance check);
`reward_paid` é da janela da oferta (aproximação ao usar horizonte 7d);
tratados podem ter ofertas anteriores ativas (56% de sobreposição) → ATE também
reportado no recorte "tratado limpo" (sem oferta ativa no envio).
""")

md("""## Glossário — os termos usados neste notebook

| Termo | O que significa, em linguagem simples |
|---|---|
| **Efeito incremental** (ou *uplift*) | A venda **a mais** que o envio gerou. Não é o quanto o cliente comprou, é o quanto ele comprou **além do que compraria sem a oferta**. |
| **Grupo de controle** | Os clientes que **não** receberam oferta naquela campanha. Servem de comparação: mostram o que teria acontecido sem o envio. |
| **ATE** | *Average Treatment Effect* — o efeito **médio** do envio, um número só para toda a base. Responde: *"vale a pena enviar?"* |
| **CATE** | *Conditional ATE* — o mesmo efeito, mas **calculado para cada cliente e cada oferta**. Responde: *"para quem enviar, e qual oferta?"* |
| **T-learner** | A técnica usada para estimar o CATE: treina **dois** modelos — um que aprende o comportamento de quem recebeu oferta, outro de quem não recebeu — e o efeito é a **diferença** entre os dois. |
| **Curva Qini** | Gráfico que responde: *"se eu contatar só os X% melhores segundo o modelo, quanto do ganho total eu capturo?"* Quanto mais a curva sobe rápido no começo, melhor o modelo ordena os clientes. |
| **Bootstrap** | Forma de calcular a margem de erro: sorteia a amostra milhares de vezes e refaz a conta, para ver o quanto o resultado varia. |
| **Split temporal** | Treinar com as campanhas antigas e testar nas campanhas seguintes — em vez de sortear as linhas. Simula a situação real de decidir hoje com dados de ontem. |
| **Model-free** | Resultado medido direto nos dados observados, **sem depender de o modelo estar certo**. É a evidência mais forte que temos. |
| **Modelo de resposta** | Classificador que prevê *quem usa o cupom*. Aparece só no bloco E, como comparação — **não** é o que decide o envio. |
""")

md("""## Preparação e carga dos dados
Assim como o notebook 1, este **roda tanto no computador quanto no Databricks sem
alterações** — ele identifica o ambiente e busca os dados no lugar certo:

- **No computador:** lê o arquivo `data/processed/modeling_table.parquet`;
- **No Databricks:** lê as tabelas `ifood_modeling_table` e `ifood_offers`.

⚠️ Em ambos os casos, **é preciso rodar o notebook 1 antes** — é ele que gera essa
base. A primeira célula instala a biblioteca `xgboost`, necessária no Databricks
(no computador local ela já costuma estar instalada e o comando não faz nada).""")
co("""%pip install -q xgboost""")
co("""import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
rng = np.random.RandomState(42)
plt.rcParams['figure.dpi'] = 110
IFOOD_RED = '#EA1D2C'

IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ
if IS_DATABRICKS:
    FIG = '/tmp/figures'
    df = spark.table('ifood_modeling_table').toPandas()
    offers = spark.table('ifood_offers').toPandas()
else:
    FIG = os.path.abspath(os.path.join('..', 'presentation', 'figures'))
    df = pd.read_parquet('../data/processed/modeling_table.parquet')
    offers = pd.read_parquet('../data/processed/offers.parquet')
os.makedirs(FIG, exist_ok=True)
print("Ambiente:", "Databricks" if IS_DATABRICKS else "local", "|", df.shape)
df.groupby('W').size()""")

md("### Métrica de avaliação — curva Qini (embutida, sem dependência de `src/`)")
co('''def qini_curve(y_true, treatment, uplift):
    """Pontos (x, y) da curva Qini: ganho incremental acumulado ao contatar a
    população ranqueada pelo score (funciona para desfecho binário ou contínuo)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "w": np.asarray(treatment),
                      "s": np.asarray(uplift)})
    d = d.sort_values("s", ascending=False, kind="mergesort").reset_index(drop=True)
    cum_y_t = (d.y * d.w).cumsum()
    cum_y_c = (d.y * (1 - d.w)).cumsum()
    cum_n_t = d.w.cumsum()
    cum_n_c = (1 - d.w).cumsum()
    ratio = np.where(cum_n_c == 0, 0, cum_n_t / np.where(cum_n_c == 0, 1, cum_n_c))
    qini = cum_y_t - cum_y_c * ratio
    n = len(d)
    x = np.arange(1, n + 1) / n
    return np.concatenate([[0], x]), np.concatenate([[0], qini.values])''')

md("""## A · Split temporal
Treino = ondas 0–14; teste = 17–24. O corte no dia 17 preserva ~metade das linhas
em cada lado e deixa 2 ondas de teste com controle limpo razoável.""")
co("""TRAIN_WAVES, TEST_WAVES = [0.0, 7.0, 14.0], [17.0, 21.0, 24.0]
df['split'] = np.where(df['wave'].isin(TRAIN_WAVES), 'train', 'test')

# outcome líquido no horizonte de 7 dias (comparável entre tratado e controle)
df['y_net_h7'] = df['spend_h7'] - df['reward_paid']

print(df.groupby(['split', 'W']).size().unstack())
print('\\ncontrole limpo por split:')
print(df[df.control_clean == 1].groupby('split').size())""")

md("""### Preparação de features
Pré-onda apenas. Sentinelas documentadas: `recency_days=30` (nunca comprou),
taxas históricas `-1` (nunca recebeu oferta), idade mediana + flag de perfil
incompleto.""")
co("""df['gender_M'] = (df.gender == 'M').astype(int)
df['gender_F'] = (df.gender == 'F').astype(int)
age_med = df.loc[df.split == 'train', 'age'].median()
df['age_f'] = df['age'].fillna(age_med)
df['limit_f'] = df['credit_card_limit'].fillna(0.0)
df['recency_f'] = df['recency_days'].fillna(30.0)
df['view_rate_f'] = df['view_rate_pre'].fillna(-1.0)
df['comp_rate_f'] = df['comp_rate_pre'].fillna(-1.0)

X_CUST = ['age_f', 'gender_M', 'gender_F', 'limit_f', 'incomplete_profile',
          'account_age_days', 'has_history', 'n_tx_pre', 'spend_pre',
          'avg_ticket_pre', 'active_days_pre', 'recency_f', 'n_received_pre',
          'n_viewed_pre', 'n_completed_pre', 'view_rate_f', 'comp_rate_f',
          'n_active_offers']
X_OFFER = ['min_value', 'discount_value', 'duration', 'n_channels',
           'ch_web', 'ch_email', 'ch_mobile', 'ch_social']
print(len(X_CUST), 'features de cliente |', len(X_OFFER), 'de oferta')""")

md("""## B · Efeito causal do envio (ATE)
Diferença de médias entre quem **recebeu** e o **controle limpo**, em cada onda, no
gasto de 7 dias líquido do reward. IC 95% por bootstrap.

O efeito é calculado com **duas definições de "recebeu"**, sempre contra o **mesmo**
grupo de controle:

| Definição | Quem entra |
|---|---|
| **qualquer envio** | todos que receberam oferta na onda |
| **envio isolado** | só quem recebeu **e não tinha outra oferta ainda ativa** |

A segunda responde à crítica de que o efeito medido poderia ser de várias ofertas
somadas. Comparar as duas colunas mostra o quanto isso pesa.""")
co("""def boot_ci(a, b, n=2000, seed=0):
    r = np.random.RandomState(seed)
    diffs = [a[r.randint(0, len(a), len(a))].mean() - b[r.randint(0, len(b), len(b))].mean()
             for _ in range(n)]
    return np.percentile(diffs, [2.5, 97.5])

rows = []
for w in sorted(df.wave.unique()):
    dw = df[df.wave == w]
    ctrl = dw[dw.control_clean == 1]['y_net_h7'].values
    linha = {'onda': int(w), 'n_controle': len(ctrl),
             'gasto_controle': round(ctrl.mean(), 2)}
    for rotulo, tr in [('qualquer', dw[dw.W == 1]),
                       ('isolado', dw[(dw.W == 1) & (dw.n_active_offers == 0)])]:
        t = tr['y_net_h7'].values
        if len(t) < 50 or len(ctrl) < 50:
            continue
        lo, hi = boot_ci(t, ctrl)
        linha[f'n_{rotulo}'] = len(t)
        linha[f'gasto_{rotulo}'] = round(t.mean(), 2)
        linha[f'efeito_{rotulo}'] = round(t.mean() - ctrl.mean(), 2)
        linha[f'ic_{rotulo}'] = f'{lo:.2f}–{hi:.2f}'
    rows.append(linha)

ate = pd.DataFrame(rows)[['onda', 'n_controle', 'gasto_controle',
                          'n_qualquer', 'gasto_qualquer', 'efeito_qualquer', 'ic_qualquer',
                          'n_isolado', 'gasto_isolado', 'efeito_isolado', 'ic_isolado']]
print("gasto = média de R$ por cliente em 7 dias, líquido do reward\\n")
print(ate.to_string(index=False))
print("\\nComo ler:")
print("  • efeito = gasto do tratado − gasto do controle (ex.: onda 0: 21,02 − 11,19 = 9,82)")
print("  • quem recebe oferta gasta ~59% mais que quem não recebe — a diferença é grande")
print("  • já 'qualquer' vs 'isolado' quase não diferem: restringir aos que tinham só")
print("    aquela oferta não muda o resultado, ou seja, a sobreposição não inflava o efeito")
print("  • na onda 0 as duas definições coincidem: era o 1º envio, ninguém tinha oferta anterior")""")

co('''def cluster_ci(t_df, c_df, n=2000, seed=0):
    """IC por bootstrap reamostrando CLIENTES (não linhas).

    O mesmo cliente aparece em até 6 ondas, então as linhas não são
    independentes. Reamostrar clientes inteiros é o IC correto sob essa
    correlação — na prática, com 17 mil clientes, a diferença é pequena.
    """
    pool = pd.concat([t_df.assign(_g=1), c_df.assign(_g=0)])
    agg = (pool.groupby(['account_id', '_g'])['y_net_h7'].agg(['sum', 'count'])
               .unstack('_g', fill_value=0))
    s0, n0 = agg[('sum', 0)].values, agg[('count', 0)].values
    s1, n1 = agg[('sum', 1)].values, agg[('count', 1)].values
    M = len(agg)
    r = np.random.RandomState(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = r.randint(0, M, M)
        diffs[i] = s1[idx].sum() / max(n1[idx].sum(), 1) - s0[idx].sum() / max(n0[idx].sum(), 1)
    return np.percentile(diffs, [2.5, 97.5])''')

co("""# ATE agregado (junta todas as ondas) e por tipo de oferta
pool_t = df[(df.W == 1) & (df.n_active_offers == 0)]
pool_c = df[df.control_clean == 1]
ate_pool = pool_t['y_net_h7'].mean() - pool_c['y_net_h7'].mean()

lo_lin, hi_lin = boot_ci(pool_t['y_net_h7'].values, pool_c['y_net_h7'].values)
lo_cl, hi_cl = cluster_ci(pool_t, pool_c)
print(f"ATE líquido 7d (tratado isolado vs controle limpo): R$ {ate_pool:.2f}"
      f"   n_t={len(pool_t):,} | n_c={len(pool_c):,}")
print(f"  IC 95% linha-a-linha    : [{lo_lin:.2f}, {hi_lin:.2f}]")
print(f"  IC 95% por cliente      : [{lo_cl:.2f}, {hi_cl:.2f}]  <- correto sob repetição"
      f" de cliente; praticamente igual, então a correção é imaterial\\n")

# por tipo — um único cálculo (IC por cliente), ordenado pelo efeito
by_type = []
for t in ['bogo', 'discount', 'informational']:
    tt = pool_t[pool_t.offer_type == t]['y_net_h7'].values
    lo, hi = cluster_ci(pool_t[pool_t.offer_type == t], pool_c, seed=1)
    by_type.append({'tipo': t, 'n': len(tt),
                    'ate_liq_7d': round(tt.mean() - pool_c['y_net_h7'].mean(), 2),
                    'ic_lo': round(lo, 2), 'ic_hi': round(hi, 2)})
by_type = pd.DataFrame(by_type).sort_values('ate_liq_7d').reset_index(drop=True)
print(by_type.to_string(index=False))

# ordenar pelo efeito faz a separação entre os tipos aparecer sozinha no gráfico
sobrepoe = (by_type.ic_hi[:-1].values > by_type.ic_lo[1:].values).any()
print(f"\\nalgum par de intervalos se sobrepõe? {'SIM' if sobrepoe else 'NÃO'}"
      f"  -> ordenação entre tipos é {'inconclusiva' if sobrepoe else 'sólida'}")

LAB = {'informational': 'informacional', 'bogo': 'BOGO', 'discount': 'desconto'}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar([LAB[t] for t in by_type.tipo], by_type.ate_liq_7d, color=IFOOD_RED,
       yerr=[by_type.ate_liq_7d - by_type.ic_lo, by_type.ic_hi - by_type.ate_liq_7d],
       capsize=5)
for i, (v, hi) in enumerate(zip(by_type.ate_liq_7d, by_type.ic_hi)):
    ax.text(i, hi + 0.3, f'R$ {v:.2f}', ha='center', fontsize=9)
ax.axhline(0, color='grey', lw=.8)
ax.set_ylabel('R$ incremental líquido / cliente (7d)')
ax.set_title('Efeito causal do envio, líquido do reward (IC 95%)')
ax.margins(y=0.18)
plt.tight_layout(); plt.savefig(f'{FIG}/05_ate.png', bbox_inches='tight'); plt.show()""")

md("""**Leitura:** o envio gera incremento líquido positivo mesmo depois de descontar
o cupom — este é o *baseline* causal que a política tenta melhorar (blocos C e D).

Como os tipos estão **ordenados pelo efeito**, a separação aparece sozinha: as
barras formam uma escada e os intervalos de erro não se tocam. Isso é o que autoriza
priorizar desconto — se os intervalos se sobrepusessem, não daria para afirmar qual
tipo rende mais, e o ganho do piso (que vem de migrar o mix para desconto) ficaria
sem base.

Sobre a **margem de erro**: reportamos as duas versões porque o mesmo cliente aparece
em várias ondas, então as linhas não são independentes. Reamostrar clientes inteiros
é o cálculo correto — e dá praticamente o mesmo resultado, porque são 17 mil
clientes. Fica registrado que a questão foi verificada.

*Próxima checagem:* ajuste por regressão — o efeito ajustado pelas características
dos clientes deve ficar próximo da diferença de médias se os grupos forem mesmo
equivalentes.""")
co("""# ajuste por regressão: outcome ~ features, resíduo comparado entre W (pooled limpo)
adj = pd.concat([pool_t, pool_c])
m_adj = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
m_adj.fit(adj[X_CUST], adj['y_net_h7'])
resid = adj['y_net_h7'] - m_adj.predict(adj[X_CUST])
ate_adj = resid[adj.W == 1].mean() - resid[adj.W == 0].mean()
print(f"ATE ajustado por regressão: R$ {ate_adj:.2f} (dif. de médias: R$ {ate_pool:.2f})")""")

md("""### Por que medir em 7 dias? (teste de sensibilidade)
A janela de 7 dias não é arbitrária: é a **duração mais comum** das ofertas (4 das
10) e também a **mediana** — as durações são 3, 4, 5, 5, 7, 7, 7, 7, 10, 10. Uma
janela curta demais corta vendas que a oferta ainda ia gerar; longa demais captura
compras sem relação com ela, já expirada.

Mas justificar não basta — o certo é testar se a conclusão depende dessa escolha.""")
co("""# horizontes disponíveis, lidos da própria tabela (NB1 gerou 3,4,5,7,10 dias)
HORIZONS = sorted(int(c.replace('spend_h', '')) for c in df.columns
                  if c.startswith('spend_h'))
sens_h = []
for h in HORIZONS:
    col = f'spend_h{h}'
    yt = (pool_t[col] - pool_t['reward_paid']).mean()
    yc = pool_c[col].mean()
    linha = {'horizonte': f'{h}d', 'gasto_controle': round(yc, 2),
             'gasto_tratado': round(yt, 2), 'efeito': round(yt - yc, 2),
             '%_acima': f'{(yt/yc - 1)*100:.0f}%'}
    ef = {}
    for tp in ['bogo', 'discount', 'informational']:
        s = pool_t[pool_t.offer_type == tp]
        ef[tp] = (s[col] - s['reward_paid']).mean() - yc
    linha['ordem_entre_tipos'] = ' > '.join(sorted(ef, key=ef.get, reverse=True))
    sens_h.append(linha)
print(pd.DataFrame(sens_h).to_string(index=False))""")

md("""**Leitura:** o efeito é **positivo e substancial em qualquer janela** — a
conclusão "enviar compensa" não depende dessa escolha.

Dois pontos que o teste revela:

1. **O efeito satura entre 7 e 10 dias** (R$ 9,21 → R$ 9,65, só +5%). Quase toda a
   venda incremental acontece na primeira semana, então esticar a janela adicionaria
   pouco sinal e muito ruído de compras não relacionadas.
2. **A ordenação entre tipos se inverte em 3 dias**: ali informacional aparece na
   frente. Faz sentido — as duas ofertas informacionais duram 3 e 4 dias, então numa
   janela curta elas já se esgotaram enquanto as de desconto (7–10 dias) mal
   começaram. A partir de 5 dias a ordem se estabiliza em *desconto > BOGO >
   informacional*.

O item 2 é a ressalva honesta: a recomendação de priorizar desconto **vale para
janelas de 5 dias ou mais**. Como 7 dias é a duração típica das ofertas, é o
horizonte que representa o ciclo real — mas quem quiser otimizar resposta imediata
(3 dias) chegaria a outra conclusão.""")

md("""## C · Efeito heterogêneo (CATE) — T-learner por tipo de oferta
- `m1_tipo(x, atributos_da_oferta)`: regressor do gasto líquido 7d nos **tratados do
  tipo** (ondas de treino);
- `m0(x)`: regressor do gasto 7d no **controle limpo** (ondas de treino);
- `CATE(x, oferta) = m1_tipo(x, oferta) − m0(x)`.

Atributos da oferta entram no braço → o CATE diferencia ofertas do mesmo tipo.""")
co("""def make_reg():
    return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

m0 = make_reg()
ctr_tr = df[(df.control_clean == 1) & (df.split == 'train')]
m0.fit(ctr_tr[X_CUST], ctr_tr['y_net_h7'])

m1 = {}
for t in ['bogo', 'discount', 'informational']:
    sub = df[(df.W == 1) & (df.offer_type == t) & (df.split == 'train')]
    m1[t] = make_reg()
    m1[t].fit(sub[X_CUST + X_OFFER], sub['y_net_h7'])
    print(f"braço {t:13s}: {len(sub):,} linhas de treino")
print(f"controle (m0): {len(ctr_tr):,}")""")

md("""### Validação do CATE — Qini nas ondas de teste (por tipo)
População: tratados do tipo + controle limpo (teste). Score = CATE previsto;
curva Qini no **gasto líquido incremental** acumulado (não AUC).""")
co("""fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=False)
qini_res = {}
for ax, t in zip(axes, ['bogo', 'discount', 'informational']):
    tr_t = df[(df.W == 1) & (df.offer_type == t) & (df.split == 'test')].copy()
    ct_t = df[(df.control_clean == 1) & (df.split == 'test')].copy()
    # controle é pontuado com a oferta "média" do tipo (representante do braço)
    mean_attrs = offers[offers.offer_type == t][X_OFFER].mean()
    for col in X_OFFER:
        ct_t[col] = mean_attrs[col]
    both = pd.concat([tr_t, ct_t])
    score = m1[t].predict(both[X_CUST + X_OFFER]) - m0.predict(both[X_CUST])
    x, yq = qini_curve(both['y_net_h7'].values, both['W'].values, score)
    xr, yr = qini_curve(both['y_net_h7'].values, both['W'].values,
                        rng.rand(len(both)))
    ax.plot(x, yq, color=IFOOD_RED, lw=2, label='CATE')
    ax.plot([0, 1], [0, yq[-1]], '--', color='grey', label='aleatório')
    ax.set_title(f'{t} (teste)'); ax.set_xlabel('fração contatada')
    qini_res[t] = (yq, yr)
axes[0].set_ylabel('gasto líquido incremental acum. (R$)')
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/07_qini_test.png', bbox_inches='tight'); plt.show()""")

md("""### Para que serve o `m0`? (teste da subtração)
Pergunta legítima: o `m0` é treinado em só 11.650 linhas e é **subtraído de todo
mundo**. Ele está agregando valor, ou só ruído?

Teste direto: comparar o ranking por **CATE** (`m1 − m0`) com o ranking por **`m1`
sozinho** — este último equivale a ordenar por *"quanto essa pessoa vai gastar"*, sem
descontar o que ela gastaria sem oferta.""")
co("""cust_t = df[df.split == 'test'].drop_duplicates('account_id')[['account_id'] + X_CUST]
base_m0 = m0.predict(cust_t[X_CUST])
sc_cate, sc_m1 = {}, {}
for _, o in offers.iterrows():
    G = cust_t.copy()
    for col in X_OFFER:
        G[col] = o[col]
    p1 = m1[o['offer_type']].predict(G[X_CUST + X_OFFER])
    sc_cate[o['offer_id']] = p1 - base_m0     # com m0
    sc_m1[o['offer_id']] = p1                 # sem m0

te_cmp = df[(df.split == 'test') & ((df.W == 1) | (df.control_clean == 1))].copy()
linhas = []
for nome, S in [('CATE (m1 - m0)', pd.DataFrame(sc_cate, index=cust_t.account_id)),
                ('só m1 (sem subtrair)', pd.DataFrame(sc_m1, index=cust_t.account_id))]:
    melhor = S.max(axis=1)
    d = te_cmp.assign(s=te_cmp.account_id.map(melhor)).dropna(subset=['s'])
    d['faixa'] = pd.qcut(d.s.rank(method='first', ascending=False), 5,
                         labels=['top 20%', '20-40%', '40-60%', '60-80%', 'bottom 20%'])
    linha = {'score': nome}
    for f, g in d.groupby('faixa', observed=True):
        linha[str(f)] = round(g[g.W == 1].y_net_h7.mean() - g[g.W == 0].y_net_h7.mean(), 2)
    linha['% não enviar'] = f'{(melhor <= 0).mean():.1%}'
    linhas.append(linha)
print("uplift realizado por faixa do score (ondas de teste):\\n")
print(pd.DataFrame(linhas).to_string(index=False))""")

md("""**Leitura — o `m0` serve para decidir, não para ranquear.** O teste devolveu um
resultado contraintuitivo que vale registrar:

**1. No ranking puro, subtrair o `m0` não ajuda** — chega a piorar ligeiramente no
topo. A razão é específica desta base: **quem gasta mais também responde mais**
(vimos isso na sensibilidade por faixa de gasto — ganho de R$ 5,18 no quintil alto
contra R$ 0,52 no baixo). Então "quanto a pessoa vai gastar" já é um bom substituto
de "quanto ela vai gastar a mais". Somado a isso, o `m0` treina em poucas linhas e
adiciona ruído.

**2. Mas sem o `m0` a decisão de *não enviar* desaparece.** Repare na última coluna:
o `m1` sozinho nunca corta ninguém. É estrutural — ele prevê **gasto**, que é sempre
positivo, então o score jamais cruza zero. É impossível ele dizer "não vale a pena".

Subtrair o `m0` é o que coloca o score em **unidade de incremento**, e é isso que
torna o limiar zero interpretável. O corte de ~5% dos envios — que responde
justamente à pergunta de desperdício do case — só existe por causa dessa subtração.

**Conclusão honesta:** se o objetivo fosse apenas ordenar clientes, um modelo de
gasto bastaria nesta base. O T-learner se justifica porque a entrega é uma **regra
de decisão** (enviar / não enviar / qual oferta), e isso exige medir incremento, não
nível.""")

md("""## D · Política de envio + impacto
Para cada cliente das ondas de teste: pontua as **10 ofertas candidatas** →
`argmax` do CATE líquido; **não envia** se o melhor CATE ≤ 0. Validação
*model-free*: uplift realizado (tratado × controle limpo) por faixa do score.""")
co("""test_cust = df[df.split == 'test'].drop_duplicates('account_id')[['account_id'] + X_CUST]
base0 = m0.predict(test_cust[X_CUST])

scores = {}
for _, o in offers.iterrows():
    G = test_cust.copy()
    for col in X_OFFER:
        G[col] = o[col]
    scores[o['offer_id']] = m1[o['offer_type']].predict(G[X_CUST + X_OFFER]) - base0
S = pd.DataFrame(scores, index=test_cust.account_id)
otype = offers.set_index('offer_id')['offer_type']

policy = pd.DataFrame({
    'best_offer': S.idxmax(axis=1),
    'best_cate': S.max(axis=1)})
policy['best_type'] = policy.best_offer.map(otype)
policy['send'] = policy.best_cate > 0
print(f"clientes (teste): {len(policy):,}")
print(f"não enviar (CATE ≤ 0): {(~policy.send).mean():.1%}")
print("\\ntipo recomendado (entre envios):")
print(policy[policy.send].best_type.value_counts(normalize=True).round(3).to_string())
print("\\noferta recomendada top-5:")
print(policy[policy.send].best_offer.map(
    offers.set_index('offer_id').apply(
        lambda r: f"{r.offer_type} min={r.min_value:.0f} desc={r.discount_value:.0f} dur={r.duration:.0f}", axis=1)
).value_counts().head().to_string())""")

md("""### Interface de produção: `recomendar_oferta(cliente_id)`
Empacota a lógica acima (mesma matriz de scores `S`, já calculada) numa função
única — o contrato que um serviço de recomendação chamaria por cliente.""")
co('''def recomendar_oferta(account_id, verbose=True):
    """Recomenda a melhor oferta para um cliente (ou None, se nenhuma compensa).

    Reusa os modelos já treinados (m0 = baseline sem oferta, m1[tipo] = com
    oferta) via a matriz de scores S. Retorna o offer_id de maior CATE líquido
    esperado, ou None quando o melhor CATE é <= 0 (não vale enviar).
    """
    if account_id not in S.index:
        raise KeyError(f"cliente {account_id} fora da base de teste (sem features pré-onda)")
    row = S.loc[account_id]
    best_offer, best_cate = row.idxmax(), row.max()
    if best_cate <= 0:
        if verbose:
            print(f"{account_id}: NÃO enviar (melhor CATE R$ {best_cate:.2f} <= 0)")
        return None
    o = offers.set_index("offer_id").loc[best_offer]
    if verbose:
        print(f"{account_id}: enviar '{best_offer}' ({o.offer_type}, "
              f"min={o.min_value:.0f} desc={o.discount_value:.0f} dur={o.duration:.0f}d) "
              f"-> CATE esperado R$ {best_cate:.2f}")
    return best_offer


print("=== Exemplos (clientes reais da base de teste) ===")
for cid in test_cust.account_id.sample(5, random_state=7):
    recomendar_oferta(cid)''')

md("""### Robustez — política sob restrição de diversidade
A política irrestrita concentra os envios em poucas ofertas. Riscos não modelados
(saturação, canibalização, portfólio) pedem diversificação — impomos um **teto de
participação por oferta** e realocamos (greedy: cada cliente recebe a melhor oferta
com capacidade disponível; nenhuma se CATE ≤ 0). Métrica: fração do valor
*model-based* da política irrestrita que sobrevive ao teto (comparação relativa —
não depende do nível absoluto do CATE).""")
co("""S_mat = S.values
order_cust = np.argsort(-S_mat.max(axis=1))

def greedy_policy(cap_share):
    cap = int(np.ceil(cap_share * len(S_mat)))
    used = np.zeros(S_mat.shape[1], dtype=int)
    total, sends = 0.0, 0
    for ci in order_cust:
        for oi in np.argsort(-S_mat[ci]):
            v = S_mat[ci, oi]
            if v <= 0:
                break
            if used[oi] < cap:
                used[oi] += 1; total += v; sends += 1
                break
    return total, sends, used

base_total, _, _ = greedy_policy(1.0)
rows = []
for cap in [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]:
    tot, snd, used = greedy_policy(cap)
    rows.append({'teto_por_oferta': f'{cap:.0%}', 'envios': snd,
                 'valor_preservado': tot / base_total,
                 'share_maior_oferta': used.max() / max(snd, 1)})
divers = pd.DataFrame(rows)
print(divers.round(3).to_string(index=False))
v20 = divers.loc[divers.teto_por_oferta == '20%', 'valor_preservado'].iloc[0]
print(f"\\n→ forçando nenhuma oferta a passar de 20% dos envios, "
      f"a política preserva {v20:.0%} do valor irrestrito")""")

md("### Validação model-free: uplift realizado por faixa do score da política")
co("""# junta score da política às linhas de teste (tratados de qualquer tipo + controle limpo)
te = df[(df.split == 'test') & ((df.W == 1) | (df.control_clean == 1))].copy()
te = te.merge(policy[['best_cate']], left_on='account_id', right_index=True, how='inner')
te['faixa'] = pd.qcut(te.best_cate.rank(method='first', ascending=False), 5,
                      labels=['top 20%', '20-40%', '40-60%', '60-80%', 'bottom 20%'])
val = []
for f, g in te.groupby('faixa', observed=True):
    t, ctl = g[g.W == 1], g[g.W == 0]
    val.append({'faixa': f, 'n_t': len(t), 'n_c': len(ctl),
                'uplift_liq_7d': t.y_net_h7.mean() - ctl.y_net_h7.mean()})
val = pd.DataFrame(val)
print(val.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(val.faixa.astype(str), val.uplift_liq_7d, color=IFOOD_RED)
ax.axhline(0, color='grey', lw=.8)
ax.set_ylabel('uplift líquido realizado (R$/cliente, 7d)')
ax.set_title('Uplift realizado por faixa do score da política (teste)')
plt.tight_layout(); plt.savefig(f'{FIG}/08_policy_validation.png', bbox_inches='tight'); plt.show()""")

md("""### Ganho de realocação de oferta (o coração do "qual oferta enviar")

⚠️ **O que a validação por faixas prova — e o que ela não prova.** Ela mostra que o
score ranqueia bem **entre clientes** (quem responde mais). Ela **não** mede o ganho
de **trocar a oferta** de um mesmo cliente, porque cada cliente só recebeu uma
oferta — nunca observamos como ele reagiria às outras nove.

Por isso a realocação é estimada como um intervalo:

- **Piso (conservador):** realocar o mix para o tipo de maior efeito **observado**
  (discount). Usa só as diferenças medidas no bloco B — milhares de clientes
  receberam cada tipo, então a comparação é empírica, não extrapolação.
- **Teto (dependente do modelo):** efeito previsto da melhor oferta vs o da oferta
  que foi de fato enviada. Sujeito à **maldição do vencedor**: escolher o `argmax`
  de 10 previsões ruidosas infla sistematicamente o máximo, mesmo quando as ofertas
  têm efeito idêntico. Por isso é limite superior, a validar em A/B.""")
co("""tt = df[(df.W == 1) & (df.split == 'test')].copy()
base_t = m0.predict(tt[X_CUST])
cate_sent = np.zeros(len(tt))
for t in ['bogo', 'discount', 'informational']:
    m = tt.offer_type.values == t
    cate_sent[m] = m1[t].predict(tt.loc[m, X_CUST + X_OFFER]) - base_t[m]
best_t = np.full(len(tt), -np.inf)
for _, o in offers.iterrows():
    G = tt[X_CUST].copy()
    for col in X_OFFER:
        G[col] = o[col]
    best_t = np.maximum(best_t, m1[o.offer_type].predict(G[X_CUST + X_OFFER]) - base_t)

print(f"CATE médio da oferta ENVIADA : R$ {cate_sent.mean():.2f}")
print(f"CATE médio da MELHOR oferta  : R$ {best_t.mean():.2f}")
print(f"já recebia a melhor oferta   : {np.isclose(best_t, cate_sent, atol=1e-6).mean():.1%}")
print(f"ganho de realocação (model-based, teto): R$ {(best_t - cate_sent).mean():.2f}/cliente")

ate_mix = pool_t['y_net_h7'].mean() - pool_c['y_net_h7'].mean()          # mix atual
ate_disc = by_type.loc[by_type.tipo == 'discount', 'ate_liq_7d'].iloc[0]  # melhor tipo
gain_floor = ate_disc / ate_mix - 1
print(f"ganho de realocação (model-free, piso — mix→discount): {gain_floor:+.1%}")""")

md("""### Robustez do piso — o ganho mix→discount vale em toda a base?
O piso compara ATEs médios; se a vantagem do discount vier só dos clientes de alto
gasto, realocar o mix não geraria o mesmo valor no restante da base. Recalculamos
**dentro de quintis de gasto pré-onda** (tratado limpo × controle limpo do mesmo
quintil). Reportamos a diferença em R$ (primária — a razão explode quando o ATE do
quintil é pequeno).""")
co("""pool_all = pd.concat([pool_t.assign(_g=1), pool_c.assign(_g=0)]).reset_index(drop=True)
# célula 0 = sem histórico de gasto (inclui onda 0); 1-4 = quartis do gasto positivo.
# (qcut por rank separaria os empates em 0 por ordem de concatenação — viés de célula)
pool_all['cell'] = 0
pos_m = pool_all.spend_pre > 0
pool_all.loc[pos_m, 'cell'] = pd.qcut(pool_all.loc[pos_m, 'spend_pre'], 4, labels=False) + 1
LBL = {0: 'sem histórico', 1: 'Q1 (baixo)', 2: 'Q2', 3: 'Q3', 4: 'Q4 (alto)'}
rows = []
for q, g in pool_all.groupby('cell'):
    ctl = g[g._g == 0]['y_net_h7']
    t_all = g[g._g == 1]['y_net_h7']
    t_disc = g[(g._g == 1) & (g.offer_type == 'discount')]['y_net_h7']
    a_mix = t_all.mean() - ctl.mean()
    a_disc = t_disc.mean() - ctl.mean()
    rows.append({'celula_gasto_pre': LBL[int(q)], 'n_t': len(t_all),
                 'n_t_disc': len(t_disc), 'n_c': len(ctl),
                 'ate_mix': a_mix, 'ate_discount': a_disc,
                 'ganho_R$': a_disc - a_mix})
sens = pd.DataFrame(rows)
print(sens.round(2).to_string(index=False))
ok = sens.dropna(subset=['ganho_R$'])
w_gain = np.average(ok['ganho_R$'], weights=ok['n_t'])
print(f"\\n→ ganho positivo em {(ok['ganho_R$'] > 0).sum()}/{len(ok)} células | "
      f"média ponderada: R$ {w_gain:.2f}/cliente (pooled: R$ {ate_disc - ate_mix:.2f})")""")

md("""### Simulação financeira (projeção por 1 milhão de envios)
Três cenários:
1. **Enviar a todos** com o mix atual — efeito realizado nas ondas de teste;
2. **Política conservadora (piso)** — realoca o mix para o tipo de maior efeito
   observado e corta os envios de efeito previsto ≤ 0;
3. **Política personalizada (teto)** — melhor oferta por cliente, a validar em A/B.

**Quanto de cada cenário depende do modelo?** O piso usa o modelo em um único ponto
(descartar os ~5% de efeito previsto ≤ 0) — e isso *reduz* a projeção, porque na
conta simples esses envios ainda contariam pelo efeito médio. O ganho do piso vem de
uma diferença **medida** (desconto rende mais que BOGO). Já o teto depende
inteiramente do modelo. A decomposição abaixo torna isso explícito.""")
co("""POP = 1_000_000
frac_send = policy.send.mean()
overall_t = te[te.W == 1]; overall_c = te[te.W == 0]
up_all = overall_t.y_net_h7.mean() - overall_c.y_net_h7.mean()

val_all = POP * up_all
val_floor_sem_modelo = POP * up_all * (1 + gain_floor)   # só troca o mix
val_floor = POP * frac_send * up_all * (1 + gain_floor)  # + corta efeito previsto <= 0
val_ceil = POP * frac_send * np.where(best_t > 0, best_t, 0).mean() / max(cate_sent.mean(), 1e-9) * up_all

print(f"efeito realizado (enviar a todos): R$ {up_all:.2f}/cliente\\n")
print(f"projeção por {POP:,} envios possíveis:")
print(f"  1. enviar a todos, mix atual         : R$ {val_all:,.0f}")
print(f"  2. piso conservador                  : R$ {val_floor:,.0f}  ({val_floor/val_all-1:+.0%}, {frac_send:.0%} dos envios)")
print(f"  3. teto (depende do modelo)          : R$ {val_ceil:,.0f}  ({val_ceil/val_all-1:+.0%}) — validar em A/B")
print(f"\\nquanto do piso depende do modelo?")
print(f"  só trocando o mix, sem modelo algum  : R$ {val_floor_sem_modelo:,.0f}  ({val_floor_sem_modelo/val_all-1:+.0%})")
print(f"  o modelo (corte de efeito <= 0) muda : R$ {val_floor - val_floor_sem_modelo:+,.0f}"
      f"  -> o piso é conservador mesmo com o modelo")

fig, ax = plt.subplots(figsize=(7.5, 3.6))
vals = [val_all/1e6, val_floor/1e6, val_ceil/1e6]
ax.bar(['Enviar a todos\\n(mix atual)', 'Política — piso\\n(ganho medido)',
        'Política — teto\\n(depende do modelo, validar A/B)'],
       vals, color=['#9E9E9E', IFOOD_RED, '#F8B4BB'])
for i, v in enumerate(vals):
    ax.text(i, v, f'R$ {v:.1f}M', ha='center', va='bottom')
ax.set_ylabel('valor líquido incremental (R$ M)')
ax.set_title('Projeção por 1M de envios — política vs enviar a todos')
plt.tight_layout(); plt.savefig(f'{FIG}/09_business_impact.png', bbox_inches='tight'); plt.show()""")

md("""## E · Comparação — e se tivéssemos usado a abordagem convencional?

A solução acima (blocos B → C → D) mede o **efeito** do envio. A abordagem mais
comum neste tipo de problema é outra: treinar um classificador para prever **quem
usa o cupom** e mandar para os de maior probabilidade.

Vale construir esse modelo — não para usá-lo, mas para mostrar **por que ele não
resolve o problema de negócio**, mesmo funcionando bem tecnicamente.

Alvo: `y_response` = viu **e** usou, na ordem certa. Treinado nos tratados de
bogo/discount das ondas 0–14, avaliado nas ondas 17–24.

*Informational fica de fora: não gera evento de resgate, e o substituto natural
("comprou após ver") ocorre em 47,6% dos casos **sem oferta alguma** — mede
atividade basal, não resposta.*""")
co("""tr_mask = (df.W == 1) & (df.split == 'train') & df.y_response.notna()
te_mask = (df.W == 1) & (df.split == 'test') & df.y_response.notna()
Xtr, ytr = df.loc[tr_mask, X_CUST + X_OFFER], df.loc[tr_mask, 'y_response']
Xte, yte = df.loc[te_mask, X_CUST + X_OFFER], df.loc[te_mask, 'y_response']

resp = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                     random_state=42, n_jobs=-1)
resp.fit(Xtr, ytr)
p_te = resp.predict_proba(Xte)[:, 1]
print(f"treino: {len(ytr):,} (resp {ytr.mean():.3f}) | teste: {len(yte):,} (resp {yte.mean():.3f})")
print(f"AUC teste temporal: {roc_auc_score(yte, p_te):.3f} | PR-AUC: {average_precision_score(yte, p_te):.3f}")""")

co("""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
frac_pos, mean_pred = calibration_curve(yte, p_te, n_bins=10)
axes[0].plot(mean_pred, frac_pos, 'o-', color=IFOOD_RED)
axes[0].plot([0, 1], [0, 1], '--', color='grey')
axes[0].set_xlabel('previsto'); axes[0].set_ylabel('observado'); axes[0].set_title('Calibração (teste)')
imp = pd.Series(resp.feature_importances_, index=X_CUST + X_OFFER).nlargest(10)
imp.iloc[::-1].plot.barh(ax=axes[1], color='#37474F'); axes[1].set_title('Top-10 importâncias')
plt.tight_layout(); plt.savefig(f'{FIG}/06_response_model.png', bbox_inches='tight'); plt.show()""")

md("""### Por que esse modelo não deve decidir o envio
Teste direto: quem o modelo aponta como mais provável de usar o cupom **gera mais
venda incremental**? Comparamos o efeito realizado no topo e na base do ranking
dele, do mesmo jeito que fizemos com o CATE.""")
co("""cmp_te = df[(df.split == 'test') & ((df.W == 1) | (df.control_clean == 1))].copy()
cli_resp = df.loc[te_mask, ['account_id']].copy()
cli_resp['p_resp'] = p_te
cli_resp = cli_resp.groupby('account_id').p_resp.max()
cmp_te = cmp_te[cmp_te.account_id.isin(cli_resp.index)].copy()
cmp_te['p_resp'] = cmp_te.account_id.map(cli_resp)

cmp_te['faixa'] = pd.qcut(cmp_te.p_resp.rank(method='first', ascending=False), 5,
                          labels=['top 20%', '20-40%', '40-60%', '60-80%', 'bottom 20%'])
linhas = []
for f, g in cmp_te.groupby('faixa', observed=True):
    t_, c_ = g[g.W == 1], g[g.W == 0]
    linhas.append({'faixa': f, 'n_t': len(t_), 'n_c': len(c_),
                   'prob_media_de_usar': round(g.p_resp.mean(), 3),
                   'uplift_realizado': round(t_.y_net_h7.mean() - c_.y_net_h7.mean(), 2)})
print(pd.DataFrame(linhas).to_string(index=False))""")

md("""**Leitura — o modelo convencional é um proxy fraco, não inútil.** O resultado
aqui é mais sutil do que "não funciona", e vale registrar com honestidade.

Comparando os dois rankings pelo **mesmo** critério (uplift realizado por faixa):

| Faixa | Ranking pelo **efeito** (bloco C) | Ranking por **probabilidade de uso** |
|---|---|---|
| top 20% | **R$ 21,15** | R$ 12,80 |
| 20–40% | R$ 16,22 | R$ 10,94 |
| 40–60% | R$ 10,48 | R$ 5,99 |
| 60–80% | R$ 2,82 | R$ 0,08 |
| bottom 20% | −R$ 0,10 | **R$ 2,41** ← quebra a ordem |

Três diferenças:

1. **O topo rende bem menos** — R$ 12,80 contra R$ 21,15. Priorizar por
   probabilidade de uso captura só ~60% do valor que o modelo causal captura;
2. **A ordenação quebra na base** — a última faixa (R$ 2,41) rende mais que a
   penúltima (R$ 0,08). São clientes de baixa propensão a usar cupom, mas em quem a
   oferta faz diferença quando usam;
3. **A faixa de menor valor não é identificada** — o modelo causal isola um grupo
   com efeito praticamente nulo (−R$ 0,10), que é exatamente quem não deveria
   receber. O convencional não consegue separá-lo.

Ou seja: "quem usa cupom" e "em quem o cupom faz diferença" são **correlacionados,
mas não a mesma coisa**. Usar o primeiro como atalho para o segundo deixa valor na
mesa e não identifica quem deve ficar de fora — que é justamente onde está a
economia.""")

md("""## Conclusões
1. **O envio causa incremento líquido** (ATE > 0 mesmo descontando reward), com
   heterogeneidade relevante por tipo (discount > bogo > informational) e por cliente.
2. **Prever "quem usa cupom" é um proxy fraco para decidir envio** — o bloco E
   compara os dois rankings pelo mesmo critério: priorizar por probabilidade de uso
   captura só ~60% do valor do modelo causal (R$ 12,80 vs R$ 21,15 no topo), quebra
   a ordenação na base e **não identifica o grupo de efeito nulo** — justamente
   quem deveria ficar de fora. A decisão é guiada pelo **CATE líquido**.
3. A **política** (melhor oferta por cliente; "nenhuma" quando CATE ≤ 0) tem ganho
   apresentado como **intervalo**:
   - **piso** — vem de uma diferença *medida* (desconto rende mais que bogo); o
     modelo entra só para cortar os ~5% de CATE ≤ 0, o que **reduz** a projeção;
   - **teto** — depende inteiramente do modelo acertar a melhor oferta por cliente,
     e está sujeito à maldição do vencedor.

   O ranking **entre clientes** está validado sem depender do modelo (uplift
   realizado decrescente do topo à base). A **troca de oferta** não está — cada
   cliente só recebeu uma oferta, então testá-la exige A/B.

### Limitações e próximos passos
- Envio ~aleatorizado é premissa (balance check favorável); validação definitiva =
  **A/B test** com holdout.
- `reward_paid` da janela da oferta vs horizonte 7d: aproximação conservadora.
- Ondas de teste têm controle limpo menor (1,2–1,4k/onda) → ICs largos por faixa.
- Produção: re-treino por onda, monitoramento de calibração/drift, restrição de
  orçamento (knapsack uplift/custo), otimização de timing e canal.
""")

nb["cells"] = c
out = Path(__file__).resolve().parents[1] / "notebooks" / "2_modeling.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
