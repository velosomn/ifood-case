# iFood · Case Técnico Data Science — Direcionamento de Cupons e Ofertas

Solução baseada em dados para decidir **qual oferta enviar para cada cliente**
(incluindo *nenhuma*), maximizando o **valor incremental líquido** — a venda
adicional que o envio causa, descontado o custo do cupom.

> **TL;DR** — Os envios históricos ocorrem em 6 ondas e ~25% da base não recebe
> nada em cada onda, formando um **grupo de controle** com características
> balanceadas (~aleatorizado). Isso permite medir o efeito **causal** do envio:
> **R$ 9,21 de venda incremental líquida por cliente** (IC 95% [8,45–9,90], 7 dias).
> Uma política que escolhe a melhor oferta por cliente projeta
> **+20% de valor incremental** sobre "enviar a todos" já no cenário conservador
> (piso model-free), com upside a validar em A/B — e mostra que **apenas 10% dos
> clientes recebiam hoje sua melhor oferta**.

---

## 🧠 Enquadramento

A pergunta *"qual oferta enviar?"* é **causal**: só vale enviar se o envio **muda
o comportamento** do cliente. Prever quem completa a oferta ranqueia quem já ia
comprar — e ~**30% dos cupons resgatados são pagos sem o cliente ter visto a
oferta antes da compra** (recompensa desperdiçada).

| Decisão | Escolha | Alternativa descartada |
|---|---|---|
| **Tratamento** | `W` = **envio** da oferta (a alavanca que o negócio controla) | "Visualizou": é pós-tratamento (mediador) → viés de seleção |
| **Controle** | Clientes **sem envio na onda e sem oferta anterior ativa** (controle limpo; balance check favorável) | Quase-experimento visto/não-visto |
| **Unidade** | **(cliente × onda)** — máx. 1 oferta por cliente por onda (verificado) | (cliente × oferta): sem controle, 56% de janelas sobrepostas |
| **Target resposta** | `y_response` = **viu E usou** (`t_view ≤ t_comp` na validade); informational: viu E transacionou após ver | `completed` puro: inclui auto-resgate (16% das instâncias) |
| **Outcome causal** | Gasto em **horizontes fixos** (3–10d) líquido do reward | Janela da oferta: sem equivalente no controle |
| **Features** | Estritamente **pré-onda** (`t <` dia do envio) | Agregados do período: vazamento temporal |
| **Validação** | **Split temporal** (treino ondas 0–14, teste 17–24) + Qini/uplift realizado | Split aleatório; AUC como métrica de política |

---

## 📁 Estrutura do repositório

```
ifood-case/
├── data/
│   ├── raw/               # offers.json, profile.json, transactions.json (download)
│   └── processed/         # modeling_table.parquet (gerado pelo NB1)
├── notebooks/
│   ├── 0_eda.ipynb               # EDA dos dados brutos
│   ├── 1_data_processing.ipynb   # PySpark (dual-mode local/Databricks): dataset unificado
│   └── 2_modeling.ipynb          # ATE, modelo de resposta, CATE e política
├── src/                   # builders dos notebooks + métricas de uplift
├── presentation/          # deck 5 slides (.pptx + .md) + figuras
├── README.md
└── requirements.txt
```

---

## ⚙️ Como executar

### Local
```bash
pip install -r requirements.txt        # requer Java p/ PySpark: conda install -c conda-forge openjdk=17

# dados
cd data/raw && curl -L -O https://data-architect-test-source.s3.sa-east-1.amazonaws.com/ds-technical-evaluation-data.tar.gz \
  && tar -xzf ds-technical-evaluation-data.tar.gz && mv ds-technical-evaluation-data/*.json . && cd ../..

# notebooks (em ordem)
python -m nbconvert --to notebook --execute --inplace notebooks/0_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/1_data_processing.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/2_modeling.ipynb
```

### Databricks (Community/Free Edition)
Os notebooks 1 e 2 são **dual-mode**: importe o repo como **Git folder** (ou os
`.ipynb` via Import) e rode com **Run all**, na ordem:
1. `1_data_processing.ipynb` — detecta o ambiente, **baixa os dados da URL
   pública do S3 automaticamente** e persiste como tabelas
   (`ifood_modeling_table`, `ifood_offers`). Compatível com serverless (sem
   `sparkContext`, sem `cache()`, escrita via `saveAsTable`);
2. `2_modeling.ipynb` — instala o xgboost via `%pip` (1ª célula) e lê as tabelas
   salvas pelo NB1 **no mesmo workspace**.

O notebook 0 (EDA) é local e lê os JSONs de `data/raw/`.

---

## 📊 Principais resultados (ondas de teste, fora do tempo)

**Dados:** 10 ofertas · 17.000 clientes · ~306k eventos · 76.277 envios em 6 ondas.

- **Grupo de controle real:** ~4,3k clientes/onda sem envio; controle limpo
  balanceado em gasto prévio, idade e perfil → envio ~aleatorizado.
- **Efeito causal do envio (ATE):** **R$ 9,21**/cliente em 7 dias, líquido do
  reward [8,45–9,90]; robusto a ajuste de regressão (8,94). Por tipo:
  **discount 11,63 > bogo 8,65 > informational 5,45**.
- **Desperdício:** 9,4% das ofertas completadas **sem visualização** + 6,9%
  vistas só depois do resgate → ~30% dos cupons pagos sem efeito na compra.
- **Modelo de resposta** (viu & usou): AUC **0,795** fora do tempo.
- **CATE/política:** validação *model-free* por faixas do score — uplift
  realizado **monotônico** (top 20%: R$ 21,15 → bottom 20%: −R$ 0,10);
  só **10%** dos clientes recebiam sua melhor oferta.
- **Projeção (1M de envios):** enviar a todos **R$ 10,3M** → política
  **R$ 12,3M (+20%, piso model-free)**; teto model-based ~R$ 29M
  (**winner's curse** explicitado — validar em A/B).

---

## 📌 Premissas e limitações

1. `age == 118` é sentinela de perfil incompleto (100% coincidente com nulos de
   gênero/limite) → flag + imputação.
2. Janela de validade = `[t_recv, t_recv + duration]`; `time_since_test_start` em dias.
3. `offer completed` = cupom usado (resgate ao atingir `min_value` na validade);
   sem visualização anterior é auto-resgate → excluído do target de resposta.
4. **Envio ~aleatorizado por onda** é premissa central (suportada pelo balance
   check); validação definitiva exige **A/B test** com holdout.
5. `reward_paid` é da janela da oferta vs outcome em 7d — aproximação conservadora.
6. Ondas de teste têm controle limpo menor (1,2–1,4k) → ICs mais largos.
7. Base é um recorte socioeconômico específico (limites R$30–120k, público maduro)
   → generalização para outros estratos requer cautela.

## 🚀 Próximos passos

1. **A/B test** com holdout global para validar ATE e o teto da personalização.
2. Alocação sob **restrição de orçamento** (knapsack uplift/custo de reward).
3. Otimização de **timing e canal** do envio.
4. Produção: re-treino por onda, monitoramento de calibração e drift, score no
   motor de campanhas de CRM.
