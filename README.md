# iFood · Case Técnico Data Science — Direcionamento de Cupons e Ofertas

Solução baseada em dados para decidir **qual oferta enviar para cada cliente**
(incluindo *nenhuma*), maximizando o **valor incremental líquido** — a venda
adicional que o envio causa, descontado o custo do cupom.

> **Em resumo** — As ofertas foram enviadas em 6 lotes, e em cada um deles **cerca
> de 25% dos clientes não receberam nada**. Esse grupo funciona como comparação: ele
> mostra o que teria acontecido sem o envio. Como os dois grupos são estatisticamente
> parecidos, dá para medir quanto o envio realmente gerou de venda:
> **R$ 9,21 a mais por cliente em 7 dias**, já descontado o custo do cupom.
>
> Enviando a oferta certa para cada pessoa, a projeção é de **+20% de retorno** em
> relação a "enviar para todos" — e isso no cenário conservador. Hoje, **apenas 10%
> dos clientes recebem a oferta que faz mais sentido para eles**.

---

## 🧠 Como o problema foi enquadrado

A pergunta *"qual oferta enviar?"* não é sobre **quem usa cupom** — é sobre **em quem
o cupom faz diferença**. São coisas diferentes: quem compra muito usa cupom de
qualquer jeito, então mirar nessas pessoas é pagar desconto por uma venda que já
aconteceria.

Os dados confirmam o problema: **~30% dos cupons resgatados foram pagos sem o
cliente ter visto a oferta antes de comprar**. O desconto saiu do caixa sem ter
influenciado nada.

| Decisão | O que foi escolhido | O que foi descartado, e por quê |
|---|---|---|
| **O que estamos medindo o efeito?** | O **envio** da oferta — é a única coisa que a empresa controla | "Ter visualizado": quem abre a notificação já é mais engajado, então a comparação ficaria enviesada. Além disso, ninguém controla se o cliente vai abrir |
| **Grupo de comparação** | Clientes que **não receberam nada naquele lote** e também não tinham oferta anterior ainda valendo | Usar todos os que não receberam: parte deles ainda estava sob efeito de uma oferta anterior |
| **Uma linha da tabela representa** | Um **cliente em um lote de envio** (cada cliente recebe no máximo 1 oferta por lote) | Uma linha por oferta: não haveria linhas de quem não recebeu, e 56% das ofertas têm prazos que se cruzam |
| **O que conta como "respondeu"** | **Viu e depois usou** o cupom, dentro do prazo — só para BOGO e discount | Só "usou": inclui quem resgatou sem nunca ter visto (16% dos envios). Para as informacionais não há alvo confiável: "comprou depois de ver" acontece 47,6% das vezes mesmo sem oferta |
| **O que mede o resultado financeiro** | **Gasto em janelas de tempo fixas** (3 a 10 dias), descontando o cupom | Gasto na validade da oferta: quem não recebeu oferta não tem validade, então não daria para comparar |
| **Informações usadas para prever** | Apenas o que aconteceu **antes** de cada envio | Usar dados do período todo: seria "prever" com informação do futuro |
| **Como o modelo é testado** | Treina nos lotes antigos, testa nos seguintes | Sortear as linhas aleatoriamente: misturaria passado e futuro e inflaria o resultado |

---

## 📁 Estrutura do repositório

```
ifood-case/
├── data/
│   ├── raw/               # offers.json, profile.json, transactions.json (download)
│   └── processed/         # modeling_table.parquet (gerado pelo NB1)
├── notebooks/
│   ├── 0_eda.ipynb               # EDA dos dados brutos
│   ├── 1_data_processing.ipynb   # PySpark: dataset unificado (roda local e no Databricks)
│   └── 2_modeling.ipynb          # efeito do envio, modelo e política de ofertas
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
Os notebooks 1 e 2 **rodam sem nenhuma alteração** tanto no computador quanto no
Databricks — eles identificam o ambiente e se adaptam sozinhos. Importe o repo como
**Git folder** (ou os `.ipynb` via Import) e rode com **Run all**, na ordem:
1. `1_data_processing.ipynb` — detecta o ambiente, **baixa os dados da URL
   pública do S3 automaticamente** e persiste como tabelas
   (`ifood_modeling_table`, `ifood_offers`). Compatível com serverless (sem
   `sparkContext`, sem `cache()`, escrita via `saveAsTable`);
2. `2_modeling.ipynb` — instala o xgboost via `%pip` (1ª célula) e lê as tabelas
   salvas pelo NB1 **no mesmo workspace**.

O notebook 0 (EDA) é local e lê os JSONs de `data/raw/`.

---

## 📊 Principais resultados

Todos os números abaixo vêm dos **lotes finais**, que o modelo nunca tinha visto.

**Base:** 10 ofertas · 17.000 clientes · ~306 mil eventos · 76.277 envios em 6 lotes.

- **O grupo de comparação existe e é confiável:** ~4,3 mil clientes por lote sem
  receber nada, com gasto anterior, idade e perfil equivalentes a quem recebeu.
- **Quanto o envio gera:** **R$ 9,21 de venda a mais por cliente** em 7 dias, já
  descontado o cupom (margem de erro: R$ 8,45 a R$ 9,90). Por tipo de oferta:
  **discount R$ 11,63 > BOGO R$ 8,65 > informacional R$ 5,45**.
- **Desperdício:** 9,4% dos cupons foram usados **sem terem sido vistos** e outros
  6,9% foram vistos só depois do resgate — ~30% do desconto pago sem efeito.
- **Modelo de resposta** (viu e usou, BOGO/discount): acerta o ranqueamento em
  **80,6%** dos casos, testado em lotes futuros.
- **O modelo separa bem quem vale a pena:** entre os 20% mais bem pontuados, o
  envio gerou **R$ 21,15** por cliente; entre os 20% piores, **nada** (−R$ 0,10).
  E hoje **só 10% das pessoas recebem a oferta mais adequada a elas**.
- **Projeção para 1 milhão de envios:** enviar para todos rende **R$ 10,3M**; com a
  política, **R$ 12,3M (+20%)** no cenário conservador. Há um potencial maior
  (~R$ 29M) se a personalização completa se confirmar, mas esse número vem do
  modelo e **precisa ser validado num teste A/B** antes de ser prometido.

---

## 📌 Premissas e limitações

1. A idade 118 não é idade real — é o valor gravado quando o cadastro está
   incompleto (coincide 100% com quem não tem gênero nem limite informados).
2. Uma oferta vale de quando é recebida até o fim do seu prazo (3 a 10 dias).
3. O evento "oferta completada" significa que o cliente atingiu o gasto mínimo e o
   desconto foi pago. Quando isso acontece **sem ele ter visto a oferta**, foi
   resgate automático — por isso esses casos não contam como resposta.
4. **A premissa mais importante:** assumimos que os envios foram feitos sem
   privilegiar nenhum perfil. Os dados sustentam isso (os grupos são equivalentes),
   mas a confirmação definitiva exige um **teste A/B** de verdade.
5. O custo do cupom é contado dentro da validade da oferta, enquanto a venda é
   medida em 7 dias — pequeno desalinhamento, que joga a favor da cautela.
6. Nos lotes finais o grupo de comparação é menor (1,2 a 1,4 mil pessoas), então a
   margem de erro aumenta.
7. Esta base tem um perfil específico (limites de R$30 a 120 mil, público mais
   maduro). Aplicar as conclusões a outros públicos exige cuidado.

## 🚀 Próximos passos

1. **Teste A/B** com um grupo separado de verdade, para confirmar o efeito medido e
   o potencial da personalização completa.
2. Distribuir os cupons respeitando um **orçamento** — priorizar quem dá mais
   retorno por real gasto.
3. Otimizar **quando** e **por qual canal** enviar.
4. Colocar em produção: retreinar a cada campanha, monitorar se o modelo continua
   acertando, e integrar ao sistema de campanhas.
