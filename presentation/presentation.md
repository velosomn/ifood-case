# iFood · Direcionamento de Cupons — Apresentação (5 slides)

> Versão em markdown do deck (`ifood_case.pptx`). Público: lideranças de negócio.

---

## Slide 1 — A oferta certa, para o cliente certo — e só quando ela gera venda nova

Medimos o **efeito real de cada envio** (venda incremental, descontado o cupom) e
transformamos isso numa **política de envio por cliente**.

*Base: 17 mil clientes · 306 mil eventos · 76 mil envios em 6 campanhas.*

---

## Slide 2 — O problema: parte do orçamento de cupons não gera venda nova

![funil](figures/10_funnel.png)

- **~30% dos cupons pagos não influenciaram a compra**: o cliente atingiu o valor
  mínimo e o cupom foi resgatado **sem que ele tivesse visto a oferta** — a compra
  aconteceria de qualquer forma.
- Enviar a mesma oferta para todos ignora quem responde e quem não.

---

## Slide 3 — Quanto vale um envio: R$ 9 de venda nova, já descontado o cupom

![efeito causal](figures/05_ate.png)

Em cada campanha, ~25% dos clientes **não receberam oferta** — um grupo de
comparação equivalente aos demais. A diferença entre os grupos é a venda que o
envio **causou**: **R$ 9,21 por cliente em 7 dias** (IC 8,45–9,90).
Desconto rende mais que BOGO e que ofertas informativas.

---

## Slide 4 — A solução: uma política por cliente — a melhor oferta, ou nenhuma

![validação](figures/08_policy_validation.png)

Testado em campanhas **futuras** (fora do treino): o grupo apontado como
prioritário gerou **R$ 21** de venda incremental por cliente; o de menor
prioridade, **zero**.

> **Hoje, só 10% dos clientes recebem a sua melhor oferta.**

---

## Slide 5 — Impacto: de R$ 10,3M para R$ 12,3M por milhão de envios

![impacto](figures/09_business_impact.png)

- **+R$ 2,0M (+20%)** por 1M de envios já no cenário **conservador**, enviando 5%
  menos cupons;
- Potencial de ~R$ 29M com personalização total — número de modelo, a **confirmar
  em teste controlado (A/B)**.

**Próximos passos:** teste A/B com grupo de controle · alocação sob orçamento ·
timing e canal · integração ao motor de campanhas.
