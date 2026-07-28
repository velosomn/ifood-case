# iFood · Case Técnico Data Science — Otimização de Distribuição de Cupons

Solução baseada em dados para decidir **qual oferta enviar para cada cliente** do
iFood, maximizando o **retorno incremental** dos cupons via **uplift modeling**.

> **TL;DR** — Enviar cupom para todo mundo desperdiça orçamento: ~10% dos cupons são
> completados por clientes que *já comprariam de qualquer forma*. Um modelo de
> uplift identifica os clientes **persuadíveis** e recomenda a melhor oferta para
> cada um. Projeção: **+R$ 1,45M de valor líquido incremental (+21%) por 1M de
> cupons, enviando 20% menos**.

---

## 🧠 Abordagem

A pergunta *"qual oferta enviar para cada cliente?"* é **causal**: só vale a pena
enviar se a oferta **muda o comportamento** do cliente. Por isso não modelamos a
propensão a completar, e sim o **efeito incremental (uplift)**:

```
uplift(cliente, oferta) = P(completa | recebeu e VIU) − P(completa | recebeu e NÃO viu)
```

Como o dataset não tem grupo de controle "sem oferta", usamos um **experimento
quase-natural** já presente nos dados: nem todo cliente que **recebe** a oferta a
**visualiza**. O grupo "não visualizou" funciona como controle (premissa
documentada; ver limitações).

| Segmento | Comportamento | Ação ótima |
|---|---|---|
| **Persuadíveis** | completam **se** virem a oferta | **enviar** |
| **Certeza (sure things)** | completam de qualquer jeito | **não enviar** (economia) |
| **Causa perdida** | não completam nem vendo | não enviar |

**Modelo:** T-learner (dois modelos tratado/controle) com XGBoost, comparado a um
S-learner. **Métricas:** curva Qini, Qini-AUC e uplift@k — as adequadas para
direcionamento (medem ganho incremental por fração da base contatada).

---

## 📁 Estrutura do repositório

```
ifood-case/
├── data/
│   ├── raw/               # offers.json, profile.json, transactions.json (baixados)
│   └── processed/         # saídas Parquet do notebook 1
├── notebooks/
│   ├── 0_eda.ipynb               # análise exploratória (Overview + variável a variável + uplift)
│   ├── 1_data_processing.ipynb   # PySpark: limpeza + dataset unificado + rótulos de uplift
│   └── 2_modeling.ipynb          # uplift modeling, avaliação e impacto de negócio
├── src/                   # helpers reutilizáveis
│   ├── config.py                 # paths + factory da SparkSession
│   ├── uplift_metrics.py         # Qini, uplift@k, uplift por decil
│   ├── build_nb1.py / build_nb2.py / build_deck.py   # geradores dos artefatos
├── presentation/
│   ├── ifood_case.pptx           # 5 slides para stakeholders
│   ├── presentation.md           # mesma apresentação em markdown
│   └── figures/                  # gráficos usados nos slides
├── requirements.txt
└── README.md
```

---

## ⚙️ Como executar

### Pré-requisitos
- **Python 3.10+**
- **Java (JDK 8/11/17)** — necessário para o PySpark. Recomendado:
  ```bash
  conda install -c conda-forge openjdk=17
  ```

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Baixar os dados
```bash
cd data/raw
curl -L -o data.tar.gz https://data-architect-test-source.s3.sa-east-1.amazonaws.com/ds-technical-evaluation-data.tar.gz
tar -xzf data.tar.gz
# deixar offers.json, profile.json e transactions.json em data/raw/
```

### 3. Rodar os notebooks (em ordem)
```bash
# EDA sobre os dados BRUTOS (independente do processing) -> gráficos inline
python -m nbconvert --to notebook --execute --inplace notebooks/0_eda.ipynb

# processamento (PySpark) -> data/processed/*.parquet
python -m nbconvert --to notebook --execute --inplace notebooks/1_data_processing.ipynb

# modelagem + figuras -> presentation/figures/*.png
python -m nbconvert --to notebook --execute --inplace notebooks/2_modeling.ipynb
```
Ou abra os notebooks no Jupyter/VS Code e execute célula a célula.
O `0_eda.ipynb` usa apenas `data/raw/*.json`; o `2_modeling.ipynb` consome a base
gerada pelo `1_data_processing.ipynb`.

### 4. (Opcional) Regerar a apresentação
```bash
python src/build_deck.py   # -> presentation/ifood_case.pptx
```

> **Nota Windows / PySpark:** as transformações são 100% em Spark; a materialização
> final é feita via `pandas`/`pyarrow` (o dataset cabe em memória) para evitar a
> dependência de `winutils.exe`/`HADOOP_HOME` que o gravador Hadoop exige no
> Windows. Também não fixamos `spark.driver.memory` (evita o relaunch da JVM que
> dispara o check de HADOOP_HOME). Roda igualmente no Databricks Community Edition.

---

## 📊 Principais resultados

**Dados:** 10 ofertas · 17.000 clientes · ~306k eventos · 76.277 ofertas enviadas.

- **Recompensa desperdiçada:** 9,4% dos cupons bogo/discount são completados **sem
  serem vistos** — o equivalente a **17% de todas as conclusões** (cliente compraria
  de qualquer forma).
- **Uplift por tipo:** desconto **+29 p.p.** vs BOGO **+8 p.p.** → priorizar desconto.
- **Modelo:** T-learner com **Qini-AUC ~1,8x** o do S-learner; **uplift@10% ≈ 30%**.
- **Política:** ~10% da base tem uplift ≤ 0 → **não deve receber** oferta.
- **Impacto (por 1M de cupons):** contatar apenas o topo do ranking de uplift gera
  **+R$ 1,45M (+21%)** de valor líquido incremental, **enviando 20% menos** cupons.

---

## 📌 Premissas e limitações

1. `age == 118` é *placeholder* de idade ausente (coincide 100% com `gender` e
   `credit_card_limit` nulos) → tratado como perfil incompleto.
2. Janela de validade da oferta = `[t_recebida, t_recebida + duration]` (em dias).
3. `time_since_test_start` está em dias (0–29.75; passos de 0.25 = 6h).
4. Tratamento = oferta **visualizada** na validade. É um quase-experimento: quem
   visualiza pode ser mais engajado (viés de seleção). Controlamos por features do
   cliente; a validação definitiva exige **A/B test com holdout sem oferta**.
5. Features comportamentais são agregadas sobre todo o período de teste — há
   vazamento residual; em produção seriam de uma janela estritamente anterior.
6. Ofertas repetidas com janelas sobrepostas podem gerar dupla atribuição de um
   evento (impacto pequeno).

---

## 🚀 Próximos passos

1. A/B test com grupo de controle real para validar o uplift fora do quase-experimento.
2. Features pré-oferta para eliminar vazamento.
3. Otimização de **timing** e **canal** do envio.
4. Alocação sob restrição de orçamento (uplift por R$ investido — problema de *knapsack*).
5. Deploy do score no motor de campanhas de CRM.
