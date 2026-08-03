"""Builds notebooks/0_eda.ipynb — EDA sobre DADOS BRUTOS."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# 0 · Análise Exploratória (EDA) — iFood Offer Optimization

**Escopo:** esta EDA é feita **exclusivamente sobre os dados brutos**
(`profile`, `offers`, `transactions`), **antes** do processing. Usamos apenas
agregações simples (contagens, distribuições, `groupby`); a **atribuição por janela
temporal** (ligar recebeu→viu→completou dentro da validade) é *processing* e está no
notebook `1_data_processing.ipynb`.

Estrutura: Overview (período, missings, tipos, cardinalidades, metadados, target) →
exploração variável a variável (volumetria × taxa × valor × estabilidade temporal) →
correlações → conclusões que orientam as próximas etapas.

Como o desfecho definitivo por oferta só é construído no processing (depende da
janela temporal), para a EDA usamos um **alvo exploratório em nível de cliente**:
*"o cliente completou ao menos uma oferta?"* — suficiente para entender o poder
discriminante das variáveis brutas.
""")

md("## Importando bibliotecas")
co("""import os, sys, warnings
sys.path.append(os.path.abspath('..'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 110
IFOOD_RED = '#EA1D2C'
from src.config import OFFERS_JSON, PROFILE_JSON, TRANSACTIONS_JSON""")

md("""## Funções úteis
`plot_bad_cbk` mostra, por categoria: **quantidade** (barras) e **taxa do evento**
(linha), além de **valor financeiro** e **taxa ponderada por valor**.
`quantiliza_com_missing` faz binning por quantis tratando missing.
`plot_stacked_bar` avalia estabilidade temporal.""")
co('''def quantiliza_com_missing(df, var, var_nova, quantil):
    """Binning por quantis; valores nulos vão para o bucket -1."""
    df1 = df.loc[~df[var].isnull()].copy()
    df2 = df.loc[df[var].isnull()].copy()
    disc = KBinsDiscretizer(n_bins=quantil, encode='ordinal', strategy='quantile', subsample=None)
    df1[var_nova] = disc.fit_transform(df1[[var]]).astype(int)
    df2[var_nova] = -1
    return pd.concat([df1, df2], axis=0).reset_index(drop=True)


def plot_bad_cbk(dataframe, var, resposta, tpv, ascending=0, x_size=13, y_size=4,
                 piso=0, teto=1.0):
    """Volumetria + taxa do evento e valor + taxa ponderada por valor, por categoria.
    resposta = alvo binário; tpv = valor financeiro."""
    d = dataframe[[var, resposta, tpv]].copy()
    d['valor_evento'] = np.where(d[resposta] == 1, d[tpv], 0.0)
    d[var] = d[var].fillna('Missing')
    g = d.groupby(var).agg(qtd=(resposta, 'size'), eventos=(resposta, 'sum'),
                           valor_evento=('valor_evento', 'sum'), valor=(tpv, 'sum')).reset_index()
    g['taxa'] = g['eventos'] / g['qtd']
    g['taxa_valor'] = g['valor_evento'] / g['valor'].replace(0, np.nan)
    g[var] = g[var].astype(str)
    if ascending == 1:
        g = g.sort_values('taxa')
    print(f'[{var}] amplitude da taxa: {g.taxa.max()-g.taxa.min():.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(x_size, y_size))
    for ax, (bar_col, line_col, ttl) in zip(
            axes, [('qtd', 'taxa', 'Qtd e taxa do evento'),
                   ('valor', 'taxa_valor', 'Valor e taxa ponderada por valor')]):
        ax.bar(g[var], g[bar_col], color='#4C72B0', alpha=.85, width=.7)
        ax.set_yticks([]); ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_title(f'{ttl} · "{var}"', fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(g[var], g[line_col], color=IFOOD_RED, marker='o', ms=3)
        ax2.set_ylim([piso, teto])
        for x, y in zip(g[var], g[line_col]):
            if pd.notna(y): ax2.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout(); plt.show()
    return g


def plot_stacked_bar(df, variable, target_variable, normalize=True):
    """Barras empilhadas (normalizado) do target por categoria — estabilidade/mix."""
    ct = pd.crosstab(df[variable], df[target_variable])
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0)
    ax = ct.plot(kind='bar', stacked=True, figsize=(13, 3.6), colormap='viridis')
    ax.set_title(f'"{target_variable}" por "{variable}"' + (' (normalizado)' if normalize else ''))
    ax.set_ylabel('Proporção' if normalize else 'Qtd'); ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=7, ncol=4); plt.tight_layout(); plt.show()


def print_correlation_matrix(df, cols, corr_max=0.75):
    """Heatmap de correlação + lista de variáveis com |corr| alta."""
    corr = df[cols].corr()
    upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high = [col for col in upper.columns if any(upper[col] >= corr_max)]
    f, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 20, as_cmap=True),
                center=0, annot=True, fmt='.2f', square=True, linewidths=.5,
                annot_kws={'size': 7}, cbar_kws={'shrink': .6})
    plt.title('Matriz de correlação'); plt.tight_layout(); plt.show()
    print('Variáveis com |correlação| >=', corr_max, ':', high)
    return high''')

md("""## Importando dados (brutos)
`registered_on` vem como inteiro `YYYYMMDD` (ANOMESDIA) no JSON. É uma **data**, não
um número — convertemos para `datetime` já na carga para não tratá-la como numérica
(evita estatísticas sem sentido no `describe`/correlação).""")
co("""profile = pd.read_json(PROFILE_JSON)
offers = pd.read_json(OFFERS_JSON)
tx = pd.read_json(TRANSACTIONS_JSON)

# registered_on: ANOMESDIA (YYYYMMDD) -> data
profile['registered_on'] = pd.to_datetime(profile['registered_on'], format='%Y%m%d')
print('profile', profile.shape, '| offers', offers.shape, '| transactions', tx.shape)
print('registered_on dtype:', profile.registered_on.dtype,
      '| de', profile.registered_on.min().date(), 'a', profile.registered_on.max().date())""")

# ---------------------------------------------------------------- PROFILE
md("""# Overview — Perfil dos clientes
Shape, tipos, missings, cardinalidades, metadados e distribuições.""")
co("""print('shape:', profile.shape)
profile.head()""")
co("""# registered_on já é datetime -> describe mostra estatísticas de data, não numéricas
profile.describe(include='all').T""")

md("### Análise de missings")
co("""miss = (profile.isna().mean()*100).round(2).sort_values(ascending=False)
print(miss)
print('\\n-> gender e credit_card_limit têm ~12.8% de nulos.')""")

md("### Tipos, cardinalidades e metadados (tabela consolidada)")
co("""meta = pd.DataFrame({'dtype': profile.dtypes.astype(str),
                     'cardinalidade': profile.nunique(),
                     'missing_%': (profile.isna().mean()*100).round(2)})
meta""")

md("### O placeholder de idade (`age == 118`)")
co("""print('clientes com age==118:', (profile.age==118).sum())
print('desses, gender nulo :', profile.loc[profile.age==118,'gender'].isna().mean())
print('desses, limite nulo :', profile.loc[profile.age==118,'credit_card_limit'].isna().mean())
print('-> 118 é placeholder de perfil incompleto (100% coincide com os nulos).')""")

md("""### O placeholder tem concentração de data de cadastro?
Pergunta: o `age==118` é um **lote específico** (ex.: uma migração de dados legada
num único dia) ou está espalhado pelo tempo, proporcional ao crescimento normal da
base? Comparamos a cadência mensal de cadastro dos dois grupos.""")
co("""p_ph = profile[profile.age == 118]
p_ok = profile[profile.age != 118]

m_ph = p_ph.registered_on.dt.to_period('M').value_counts(normalize=True).sort_index()
m_ok = p_ok.registered_on.dt.to_period('M').value_counts(normalize=True).sort_index()
m_ph.index = m_ph.index.astype(str); m_ok.index = m_ok.index.astype(str)

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(m_ok.index, m_ok.values*100, color='#4C72B0', marker='.', ms=3, label=f'age != 118 (n={len(p_ok):,})')
ax.plot(m_ph.index, m_ph.values*100, color=IFOOD_RED, marker='.', ms=3, label=f'age == 118 (n={len(p_ph):,})')
ax.set_xticks(ax.get_xticks()[::3]); ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.set_ylabel('% dos cadastros do grupo, por mês')
ax.set_title('Cadência de cadastro: placeholder (age==118) vs demais — mesmo formato de curva?')
ax.legend(); plt.tight_layout(); plt.show()

max_day_ph = p_ph.registered_on.dt.date.value_counts().max()
max_day_ok = p_ok.registered_on.dt.date.value_counts().max()
print(f'maior concentração num único dia — placeholder: {max_day_ph} ({max_day_ph/len(p_ph):.1%} do grupo) '
      f'| demais: {max_day_ok} ({max_day_ok/len(p_ok):.1%} do grupo)')
print(f'dias distintos de cadastro — placeholder: {p_ph.registered_on.dt.date.nunique()} de {len(p_ph)} '
      f'| demais: {p_ok.registered_on.dt.date.nunique()} de {len(p_ok)}')

print('\\ndistribuição por ano (% dentro de cada grupo):')
print(pd.crosstab(profile.registered_on.dt.year, profile.age == 118, normalize='columns')
      .rename(columns={False: 'age!=118', True: 'age==118'}).round(3).to_string())""")
md("""**Leitura:** **não há concentração temporal.** As duas curvas mensais praticamente
se sobrepõem, e a distribuição por ano é quase idêntica entre os grupos (ex.: 2017
concentra 40,0% do grupo placeholder vs 37,8% dos demais; 2016: 23,1% vs 20,4%). O
dia de maior volume dentro do grupo placeholder representa só ~0,5% dele — muito
longe de indicar um lote/importação pontual.

Isso **descarta a hipótese de migração/evento único** como origem do placeholder.
O padrão sugere, em vez disso, uma **falha estrutural persistente** no fluxo de
cadastro — uma fração constante de usuários (~12,8%) não preenche idade/gênero/
limite, e isso se repete de forma proporcional ao longo de toda a janela de dados
(2013–2018), não em um pico isolado.""")

md("""### Distribuições — idade, gênero, limite de crédito, ano de cadastro
Dados **brutos, sem pré-filtro**: se há placeholder ou nulo, o gráfico deve
deixar isso visível (não escondemos nada antes de plotar).""")
co("""fig, ax = plt.subplots(2, 2, figsize=(12, 7))

# idade SEM filtrar o placeholder — o pico em 118 tem que aparecer no histograma
ax[0,0].hist(profile.age, bins=40, color='#4C72B0')
ax[0,0].axvline(118, color=IFOOD_RED, ls='--', lw=1.2)
ax[0,0].text(118, ax[0,0].get_ylim()[1]*0.92, ' placeholder\\n 118', color=IFOOD_RED, fontsize=8)
ax[0,0].set_title('Idade (bruta — sem excluir 118)')

# gênero: nulo vira sua própria barra ('unknown'), não desaparece
profile.gender.fillna('unknown').value_counts().plot.bar(ax=ax[0,1], color=IFOOD_RED)
ax[0,1].set_title('Gênero (unknown = nulo)'); ax[0,1].tick_params(rotation=0)

# limite: histograma só aceita número (NaN não entra), mas o nº de nulos fica anotado no título
n_null_lim = profile.credit_card_limit.isna().sum()
ax[1,0].hist(profile.credit_card_limit.dropna(), bins=30, color='#4C72B0')
ax[1,0].set_title(f'Limite do cartão ({n_null_lim} nulos = {n_null_lim/len(profile):.1%}, fora do histograma)')

if not pd.api.types.is_datetime64_any_dtype(profile['registered_on']):
    profile['registered_on'] = pd.to_datetime(profile['registered_on'], format='%Y%m%d')
profile.registered_on.dt.year.value_counts().sort_index().plot.bar(ax=ax[1,1], color='#37474F')
ax[1,1].set_title('Ano de cadastro'); ax[1,1].tick_params(rotation=0)
plt.tight_layout(); plt.show()""")
md("""**Leitura:** o histograma de idade expõe o próprio problema — um pico
isolado em **118**, destacado no gráfico, claramente fora da distribuição normal
que vai até ~100. Gênero mostra a barra `unknown` (nulo) do tamanho real (12,8%).
Limite de crédito: o histograma matemático não plota `NaN`, então o título
declara quantos ficaram de fora — os mesmos 12,8% do gênero. As três pistas
juntas (idade 118 + gênero nulo + limite nulo) confirmam que é o mesmo grupo de
perfis incompletos.""")

md("""### Idade por gênero
Densidades sobrepostas para comparar o **formato** das distribuições (não só a
média). O grupo `unknown` não entra aqui por construção: gênero nulo ≡ `age==118`,
ou seja, esse grupo não tem **nem** gênero **nem** idade válida para plotar.""")
co("""age_ok = profile[profile.age != 118]
cores = {'M': '#4C72B0', 'F': IFOOD_RED, 'O': '#2E7D32'}

fig, ax = plt.subplots(figsize=(8, 4.2))
for g in ['M', 'F', 'O']:
    s = age_ok.loc[age_ok.gender == g, 'age']
    sns.kdeplot(s, ax=ax, fill=True, alpha=.30, lw=1.8,
                color=cores[g], label=f'{g} (n={len(s):,})')
ax.set_xlabel('idade'); ax.set_ylabel('densidade')
ax.set_title('Distribuição de idade por gênero')
ax.legend()
plt.tight_layout(); plt.show()

print(age_ok.groupby('gender').age.agg(['count','mean','median','std']).round(1).to_string())
print('\\n% dentro de cada gênero por faixa etária:')
faixa = pd.cut(age_ok.age, [17,30,45,60,75,110], labels=['18-30','31-45','46-60','61-75','75+'])
print((pd.crosstab(age_ok.gender, faixa, normalize='index')*100).round(1).to_string())""")
md("""**Leitura:** as duas curvas principais têm o **mesmo pico** (~55–58 anos), mas
formatos diferentes na cauda jovem: os homens têm um **ombro visível entre 20 e 40
anos** que praticamente não existe nas mulheres — 14,3% dos homens estão na faixa
18–30, contra 7,8% das mulheres. O efeito líquido é uma diferença de **~5 anos na
média** (F 57,5 vs M 52,1) e de 5 anos na mediana (58 vs 53). O grupo `O` (n=212)
fica entre os dois, mas com amostra pequena demais para leitura conclusiva.

Isso conversa com o achado de limite de crédito: as mulheres da base são **mais
velhas e com limite maior** — consistente com renda acumulada ao longo da vida, e
não com dois efeitos independentes.""")

md("""### Idade da conta (`account_age`)
Feature de perfil = **há quanto tempo o cliente tem conta** (tempo desde
`registered_on`). Como a data absoluta do teste não é dada, usamos como referência o
**cadastro mais recente da base** (proxy do início do teste, t=0). Criamos
`account_age_days` e `account_age_years`.""")
co("""# robusto à ordem de execução: garante registered_on como datetime
if not pd.api.types.is_datetime64_any_dtype(profile['registered_on']):
    profile['registered_on'] = pd.to_datetime(profile['registered_on'], format='%Y%m%d')
ref_date = profile.registered_on.max()   # proxy de 'hoje' / t=0 do teste
profile['account_age_days'] = (ref_date - profile.registered_on).dt.days
profile['account_age_years'] = (profile['account_age_days'] / 365.25).round(2)
print('data de referência (t=0):', ref_date.date())
print(profile['account_age_years'].describe(percentiles=[.05,.25,.5,.75,.95]).round(2).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].hist(profile.account_age_years, bins=30, color='#4C72B0')
ax[0].set_title('Idade da conta (anos)'); ax[0].set_xlabel('account_age_years')
profile.registered_on.dt.year.value_counts().sort_index().plot.bar(ax=ax[1], color='#37474F')
ax[1].set_title('Ano de cadastro'); ax[1].tick_params(rotation=0)
plt.tight_layout(); plt.show()""")
md("""**Leitura:** a idade da conta vai de ~0 a ~5 anos (cadastros de 2013 a 2018),
concentrada em contas **novas (0–1 ano)** — reflexo do pico de aquisição em 2017–2018.
Poucas contas antigas (2013–2015). É um sinal de **maturidade/relacionamento** do
cliente, complementar à idade da pessoa.""")

md("""## Perfil socioeconômico do público
O `credit_card_limit` é o melhor marcador de estrato social disponível. Analisamos
sua distribuição, os cortes por faixa e — em destaque — a **combinação gênero ×
limite**.""")
co("""# Boxplot/KDE por gênero (abaixo) não plotam NaN — esse filtro é uma
# necessidade matemática do gráfico, não uma limpeza escondida. Por isso
# imprimimos exatamente quem fica de fora, E por gênero, antes de seguir —
# para provar que 'O' (other) não é afetado (age==118 não tem esse gênero).
n_excl = (profile.age == 118).sum()
print(f'excluídos desta seção: {n_excl} clientes ({n_excl/len(profile):.1%}) '
      f'— mesmo grupo de age==118 / limite nulo / gênero nulo já visto acima')
print('\\ncomposição de gênero ANTES do filtro:')
print(profile.gender.fillna('unknown').value_counts().to_string())
pv = profile[profile.age != 118].copy()
print('\\ncomposição de gênero DEPOIS do filtro (deve preservar F/M/O intactos):')
print(pv.gender.value_counts().to_string())
print('limite — describe (apenas quem tem limite conhecido):')
print(pv.credit_card_limit.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(0).to_string())
bins=[0,30000,50000,80000,100000,120000,1e9]
labels=['<30k','30-50k','50-80k','80-100k','100-120k','>120k']
pv['faixa_limite']=pd.cut(pv.credit_card_limit,bins,labels=labels)
print('\\ndistribuição por faixa de limite:')
print(pv.faixa_limite.value_counts().sort_index().to_string())
print('\\n-> piso rígido em 30k e teto em 120k: sem baixa renda e sem elite (base curada/sintética).')""")

md("""### Gênero × Limite de crédito (destaque)
`unknown` não aparece nos gráficos (a) e (b) por uma razão estrutural, não por
filtro escondido: é o mesmo grupo do placeholder `age==118`, e **100% desse
grupo também tem `credit_card_limit` nulo** — não existe valor numérico para
desenhar num boxplot ou numa densidade. O gráfico (c) inclui `unknown` de
propósito: ali ele aparece como **100% "sem dado"**, o que é informativo.""")
co("""fig, ax = plt.subplots(1, 3, figsize=(14, 4))
order = ['F','M','O']

# (a) boxplot do limite por gênero — 'unknown' fica fora: seu limite é 100% nulo
data = [pv.loc[pv.gender==g,'credit_card_limit'].dropna() for g in order]
ax[0].boxplot(data, tick_labels=order, showmeans=True)
ax[0].set_title('Limite por gênero (boxplot)'); ax[0].set_ylabel('credit_card_limit')

# (b) densidade do limite por gênero — mesma razão
for g,c in zip(order, [IFOOD_RED,'#4C72B0','#2E7D32']):
    pv.loc[pv.gender==g,'credit_card_limit'].plot.kde(ax=ax[1], label=g, color=c)
ax[1].set_title('Distribuição do limite por gênero'); ax[1].set_xlim(20000,130000); ax[1].legend()

# (c) mix de faixas de limite — aqui SIM incluímos 'unknown', usando a base
# bruta (profile, não pv) e tratando limite nulo como sua própria categoria
prof_mix = profile.copy()
prof_mix['gender'] = prof_mix['gender'].fillna('unknown')
prof_mix['faixa_limite'] = pd.cut(prof_mix.credit_card_limit, bins, labels=labels)
prof_mix['faixa_limite'] = prof_mix['faixa_limite'].astype('object').fillna('sem dado')
order_c = order + ['unknown']
mix = pd.crosstab(prof_mix.gender, prof_mix.faixa_limite, normalize='index').loc[order_c]
mix.plot(kind='bar', stacked=True, ax=ax[2], colormap='viridis')
ax[2].set_title("Faixa de limite por gênero\\n('unknown' = 100% sem dado)")
ax[2].tick_params(axis='x', rotation=0)
ax[2].legend(fontsize=7, ncol=2, title='faixa')

fig.text(0.5, -0.04,
         "(a) e (b) não incluem 'unknown' (12,8% da base): mesmo grupo do placeholder\\n"
         "age=118 — sem credit_card_limit conhecido, não há valor a plotar.",
         ha='center', fontsize=9, color=IFOOD_RED)
plt.tight_layout(); plt.show()

print('mediana de limite por gênero (unknown fica de fora — não tem limite):')
print(pv.groupby('gender').credit_card_limit.agg(['median','mean','count']).round(0).to_string())""")

md("### Gênero × Limite × Idade (mediana de limite)")
co("""pv['faixa_idade']=pd.cut(pv.age,[17,30,45,60,75,110],labels=['18-30','31-45','46-60','61-75','75+'])
heat = pv.pivot_table(index='faixa_idade', columns='gender', values='credit_card_limit', aggfunc='median')
fig, ax = plt.subplots(figsize=(6,3.6))
sns.heatmap(heat[order], annot=True, fmt='.0f', cmap='YlGnBu', ax=ax, cbar_kws={'label':'limite mediano'})
ax.set_title('Limite mediano por idade × gênero'); plt.tight_layout(); plt.show()""")

md("""### Composição geracional (quem é o público)
Descritivo — traduz a idade em gerações (nascimento ≈ 2018 − idade, dado que o teste
ocorre por volta de 2018).""")
co("""pv['nasc'] = 2018 - pv['age']
def geracao(y):
    if y <= 1964: return 'Boomer+ (≤1964)'
    if y <= 1980: return 'Gen X (1965-80)'
    if y <= 1996: return 'Millennial (1981-96)'
    return 'Gen Z (1997+)'
order_ger = ['Boomer+ (≤1964)','Gen X (1965-80)','Millennial (1981-96)','Gen Z (1997+)']
pv['geracao'] = pd.Categorical(pv['nasc'].apply(geracao), categories=order_ger, ordered=True)
comp = (pv.geracao.value_counts(normalize=True).reindex(order_ger)*100).round(1)
print(comp.to_string())
fig, ax = plt.subplots(figsize=(7,3))
comp.plot.bar(ax=ax, color=['#37474F','#4C72B0','#8E9BAE','#C9CDD3'])
ax.set_ylabel('% da base'); ax.set_title('Composição geracional do público'); ax.tick_params(rotation=0)
for i,v in enumerate(comp): ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout(); plt.show()""")
md("""**Leitura:** ~**81% Boomer + Gen X** — público maduro. Millennials são minoria
(~15%) e Gen Z quase inexistente (~3%). Coerente com o estrato financeiro estável
(limites mais altos, renda acumulada).""")

md("""**Leitura — perfil socioeconômico:**
- **Classe média a média-alta, 100% bancarizada.** Piso rígido de R$30k (sem baixa
  renda) e teto de R$120k (sem elite): a base é o "miolo" da pirâmide, provavelmente
  **filtrada/sintética** → usar o limite como **proxy ordinal de estrato**, não valor literal.
- **Público maduro** (mediana 55 anos) e financeiramente estabelecido.
- **Gênero × limite:** as **mulheres têm limite consistentemente maior** que os homens
  (mediana ~71k vs ~59k), diferença que se mantém em quase todas as faixas etárias
  (heatmap). O grupo `O` é pequeno (~1,4%) — ler com cautela.
- **Limite cresce com a idade** em ambos os gêneros (acúmulo de renda ao longo da vida).

**Implicação para o case:** os achados valem **para este estrato**; extrapolar para
classes C/D ou não-bancarizados (público real do iFood) é arriscado — registrado como
**limitação de generalização**. Público mais velho e de renda estável também tende a
uma **compra basal alta** (característica relevante a considerar na estratégia).""")

# ---------------------------------------------------------------- OFFERS
md("""# Overview — Ofertas
10 ofertas de 3 tipos. Tipo, canais, valor mínimo, desconto e duração.""")
co("""display(offers)
print('tipos:', offers.offer_type.value_counts().to_dict())""")
co("""from itertools import chain
ch = pd.Series(list(chain.from_iterable(offers.channels)))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
offers.offer_type.value_counts().plot.bar(ax=ax[0], color=IFOOD_RED); ax[0].set_title('Tipos de oferta'); ax[0].tick_params(rotation=0)
ch.value_counts().plot.bar(ax=ax[1], color='#4C72B0'); ax[1].set_title('Presença por canal'); ax[1].tick_params(rotation=0)
offers.groupby('offer_type')['discount_value'].mean().plot.bar(ax=ax[2], color='#37474F'); ax[2].set_title('Desconto médio por tipo'); ax[2].tick_params(rotation=0)
plt.tight_layout(); plt.show()""")
md("""**Leitura:** 4 BOGO, 4 discount, 2 informational; `email` está em quase todas;
`informational` não tem desconto (só divulga); duração de 3 a 10 dias — parâmetro-chave
da janela de atribuição usada no processing.""")

md("""## Mecânica das ofertas e economia (custo por conclusão)
Os três campos definem as "regras" de cada oferta:
- **`min_value`** — gasto mínimo para ativar a oferta (limiar de receita);
- **`duration`** — validade em dias (define a **janela de atribuição** do processing);
- **`discount_value`** — valor do benefício = **recompensa que o iFood paga** por conclusão.

A relação entre eles muda por tipo e define a **eficiência financeira** da oferta:
`min_value / discount_value` ≈ quanto de gasto é destravado por real de recompensa.""")
co("""mec = offers.copy()
mec['custo_recompensa'] = mec['discount_value']            # o que o iFood paga
mec['gasto_minimo'] = mec['min_value']                     # receita mínima destravada
mec['eficiencia (min/desc)'] = np.where(mec['discount_value']>0,
                                        mec['min_value']/mec['discount_value'], np.nan)
print(mec[['offer_type','min_value','discount_value','duration','eficiencia (min/desc)']]
      .sort_values('offer_type').to_string(index=False))
print('\\n=== média por tipo ===')
print(mec.groupby('offer_type').agg(
        min_value=('min_value','mean'), discount_value=('discount_value','mean'),
        duration=('duration','mean'), eficiencia=('eficiencia (min/desc)','mean')).round(2).to_string())""")
co("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
g = mec.groupby('offer_type')[['min_value','discount_value']].mean()
g.plot.bar(ax=ax[0], color=['#2E7D32', IFOOD_RED]); ax[0].set_title('Gasto mínimo vs recompensa (média)')
ax[0].tick_params(rotation=0); ax[0].legend(['min_value (receita destravada)','discount_value (custo)'], fontsize=8)
eff = mec.groupby('offer_type')['eficiencia (min/desc)'].mean().dropna()
eff.plot.bar(ax=ax[1], color='#37474F'); ax[1].set_title('Eficiência: R$ destravado por R$ de recompensa')
ax[1].tick_params(rotation=0)
for i,v in enumerate(eff): ax[1].text(i, v, f'{v:.1f}x', ha='center', va='bottom')
plt.tight_layout(); plt.show()""")
md("""**Leitura — economia das ofertas:**
- **BOGO** paga alto: `min_value ≈ discount_value` (ex.: gaste 10, ganhe 10) →
  eficiência ~**1x** (paga R$1 de recompensa por R$1 destravado).
- **Discount** é enxuto: `min_value > discount_value` (ex.: gaste 20, ganhe 5) →
  eficiência ~**4x** — destrava muito mais gasto por real de recompensa.
- **Informational** não tem `min_value` nem `discount_value` (só divulga); por isso
  seu desfecho precisa ser medido de forma diferente (transação na janela, não
  "conclusão").

Ou seja, os tipos de oferta diferem fortemente em **custo por conclusão** — fator a
levar em conta ao comparar sua rentabilidade.""")

# ---------------------------------------------------------------- TRANSACTIONS
md("""# Overview — Transações
Normalizamos o campo `value` e exploramos eventos, período e valores.""")
co("""tx.head()""")
md("""Essa tabela contém eventos de **transações**, **ofertas recebidas**, **ofertas
visualizadas** e **ofertas completadas** — todos empilhados na mesma estrutura, com
o detalhe de cada um aninhado no campo `value`.""")

md("### Distribuição relativa dos tipos de evento")
co("""tx.event.value_counts(normalize=True).plot.bar(color=IFOOD_RED, figsize=(6,3.6))""")

md("### Parsing do campo `value` e cobertura temporal")
co("""tx['offer_id'] = tx['value'].apply(lambda d: d.get('offer id') or d.get('offer_id'))
tx['amount'] = tx['value'].apply(lambda d: d.get('amount'))
print('eventos:', tx.event.value_counts().to_dict())
print('período (dias):', tx.time_since_test_start.min(), '->', tx.time_since_test_start.max())
print('clientes distintos:', tx.account_id.nunique())""")
co("""fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))

# Eventos por dia. 'offer received' é EM LOTE (só em 6 dias) -> barras;
# viewed/completed/transaction fluem no tempo -> linhas. fillna(0) evita a
# 'linha invisível' de pontos isolados entre NaNs.
byday = tx.groupby([tx.time_since_test_start.astype(int), 'event']).size().unstack('event').fillna(0)
ax[0].bar(byday.index, byday['offer received'], color='#FFB300', alpha=.55, label='offer received (lote)')
for ev, col in [('offer viewed','#2E7D32'),('offer completed','#1565C0'),('transaction',IFOOD_RED)]:
    ax[0].plot(byday.index, byday[ev], color=col, marker='o', ms=2, label=ev)
ax[0].set_title('Eventos por dia'); ax[0].set_xlabel('dia'); ax[0].legend(fontsize=6)

amt = tx.amount.dropna()
ax[1].hist(amt[amt < amt.quantile(.99)], bins=40, color='#37474F'); ax[1].set_title('Valor da transação (<p99)')
plt.tight_layout(); plt.show()
send_days = list(byday.index[byday['offer received']>0])
print('dias de envio (offer received > 0):', send_days)
print('amount: média %.2f | mediana %.2f | max %.2f' % (amt.mean(), amt.median(), amt.max()))""")
md("""**Leitura:** o **recebimento de ofertas é em lote**, concentrado em 6 dias
(**0, 7, 14, 17, 21, 24**) — não é contínuo (por isso vira barras, não linha). A
cadência **acelera** ao longo do teste: começa semanal (0→7→14) e passa a cada 3–4
dias (14→17→21→24). Cada envio dispara uma **onda decrescente** de visualizações e
conclusões nos dias seguintes. Transações ocorrem todos os dias; o valor é fortemente
assimétrico à direita (poucos tickets muito altos).""")

md("""# Funil agregado de ofertas (contagem de eventos)
Sem atribuição por janela — apenas contagens brutas por tipo de evento e por oferta.""")
co("""ev = tx.groupby('event').account_id.count()
rec, vie, com = ev['offer received'], ev['offer viewed'], ev['offer completed']
print(f'recebidas={rec:,} | vistas={vie:,} | completadas={com:,}')
print(f'taxa de visualização (agregada): {vie/rec:.1%}')
print(f'taxa de conclusão   (agregada): {com/rec:.1%}')
fig, ax = plt.subplots(figsize=(6,3))
ax.bar(['recebidas','vistas','completadas'], [rec,vie,com], color=['#4C72B0','#4C72B0','#2E7D32'])
for i,v in enumerate([rec,vie,com]): ax.text(i, v, f'{v:,}', ha='center', va='bottom')
ax.set_title('Funil agregado de ofertas'); plt.tight_layout(); plt.show()""")

md("### Funil por tipo de oferta")
co("""om = offers.set_index('id')['offer_type'].to_dict()
tx['offer_type'] = tx['offer_id'].map(om)
ETAPAS = ['offer received', 'offer viewed', 'offer completed']
tab = (tx[tx.event.isin(ETAPAS)]
       .pivot_table(index='offer_type', columns='event', values='account_id',
                    aggfunc='count', fill_value=0)
       .reindex(columns=ETAPAS, fill_value=0)
       .loc[['bogo', 'discount', 'informational']])

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(tab)); w = 0.26
cores = ['#4C72B0', '#2E7D32', IFOOD_RED]
for k, (etapa, cor) in enumerate(zip(ETAPAS, cores)):
    b = ax.bar(x + (k-1)*w, tab[etapa], w, label=etapa, color=cor)
    ax.bar_label(b, fmt='%d', fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(tab.index)
ax.set_xlabel('offer_type'); ax.set_ylabel('nº de eventos')
ax.set_title('Funil por tipo de oferta (contagem bruta de eventos)')
ax.legend(title='event', fontsize=8)
ax.margins(y=0.12)
plt.tight_layout(); plt.show()

vr = tab['offer viewed'] / tab['offer received']
cr = tab['offer completed'] / tab['offer received']
for t in tab.index:
    print(f'{t:14s}: visualização {vr[t]:5.1%} | conclusão {cr[t]:5.1%}')""")
md("""**Leitura:** a barra de conclusão **não existe** para `informational` — é a
evidência visual de que esse tipo não gera evento de resgate (sem gasto mínimo, sem
desconto). Entre os outros dois, o padrão se inverte: **bogo é mais visto** (83,4%
vs 70,2%) mas **discount é mais completado** (58,6% vs 51,4%).

Esse cruzamento é a primeira pista do achado central: se discount converte mais
apesar de ser menos visto, parte das conclusões **não depende da visualização** —
ou seja, há resgate sem que a oferta tenha influenciado a compra. A quantificação
exata, usando a janela de validade e a ordem dos eventos, vem na seção seguinte.""")

md("""## Recompensa desperdiçada: completou depois de ver, ou sem nunca ter visto?
Recorte em **bogo e discount** (informational não tem evento de conclusão). Aqui
usamos a **janela de validade** (`[t_recv, t_recv + duration]`) para ligar cada
oferta recebida às suas visualizações e conclusões, e a **ordem** entre elas.

A distinção é o coração do case: `offer completed` dispara quando o cliente atinge o
gasto mínimo na validade — **mesmo que nunca tenha visto a oferta**. Nesse caso o
desconto sai do caixa sem ter influenciado a compra.""")
co("""dur_map = offers.set_index('id')['duration'].to_dict()
typ_map = offers.set_index('id')['offer_type'].to_dict()

rec = (tx[tx.event == 'offer received'][['account_id','offer_id','time_since_test_start']]
       .rename(columns={'time_since_test_start':'t_recv'}).reset_index(drop=True))
rec['rid'] = rec.index
rec['otype'] = rec.offer_id.map(typ_map)
rec['t_end'] = rec.t_recv + rec.offer_id.map(dur_map)
vie = (tx[tx.event == 'offer viewed'][['account_id','offer_id','time_since_test_start']]
       .rename(columns={'time_since_test_start':'t_view'}))
com = (tx[tx.event == 'offer completed'][['account_id','offer_id','time_since_test_start']]
       .rename(columns={'time_since_test_start':'t_comp'}))

# primeira visualização / conclusão de cada oferta recebida, dentro da validade
mv = rec.merge(vie, on=['account_id','offer_id'])
mv = mv[(mv.t_view >= mv.t_recv) & (mv.t_view <= mv.t_end)]
mc = rec.merge(com, on=['account_id','offer_id'])
mc = mc[(mc.t_comp >= mc.t_recv) & (mc.t_comp <= mc.t_end)]
rec['first_view'] = rec.rid.map(mv.groupby('rid').t_view.min())
rec['first_comp'] = rec.rid.map(mc.groupby('rid').t_comp.min())

bd = rec[rec.otype.isin(['bogo','discount'])].copy()
bd['completou'] = bd.first_comp.notna()
bd['viu'] = bd.first_view.notna()
bd['apos_ver']   = bd.viu & bd.completou & (bd.first_view <= bd.first_comp)
bd['viu_depois'] = bd.viu & bd.completou & (bd.first_view >  bd.first_comp)
bd['sem_ver']    = bd.completou & ~bd.viu

n, nc = len(bd), bd.completou.sum()
print(f'ofertas bogo/discount recebidas: {n:,} | completadas: {nc:,} ({nc/n:.1%})\\n')
for lbl, col in [('completadas APÓS visualizar', 'apos_ver'),
                 ('completadas SEM visualizar', 'sem_ver'),
                 ('completadas e vistas SÓ DEPOIS', 'viu_depois')]:
    v = bd[col].sum()
    print(f'{lbl:32s}: {v:6,} ({v/n:5.1%} das recebidas | {v/nc:5.1%} das completadas)')""")

md("### Quebra por tipo de oferta")
co("""linhas = []
for t in ['bogo','discount']:
    s = bd[bd.otype == t]
    linhas.append({'tipo': t, 'recebidas': len(s), 'completadas': s.completou.sum(),
                   'taxa_conclusao': s.completou.mean(),
                   'apos_ver': s.apos_ver.sum(), 'sem_ver': s.sem_ver.sum(),
                   'viu_depois': s.viu_depois.sum(),
                   '%_sem_ver_das_compl': s.sem_ver.sum()/s.completou.sum()})
tab = pd.DataFrame(linhas)
print(tab.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
# (a) composição das conclusões por tipo
comp_mix = tab.set_index('tipo')[['apos_ver','viu_depois','sem_ver']]
(comp_mix.div(comp_mix.sum(axis=1), axis=0)*100).plot.bar(
    stacked=True, ax=ax[0], color=['#2E7D32','#C62828', IFOOD_RED])
ax[0].set_title('Composição das conclusões (%)'); ax[0].set_ylabel('% das conclusões')
ax[0].tick_params(rotation=0); ax[0].legend(fontsize=7)

# (b) volumes absolutos sobre o total de recebidas
x = np.arange(len(tab)); w = 0.25
ax[1].bar(x-w, tab.apos_ver, w, label='após ver', color='#2E7D32')
ax[1].bar(x,   tab.viu_depois, w, label='viu só depois', color='#C62828')
ax[1].bar(x+w, tab.sem_ver, w, label='sem ver', color=IFOOD_RED)
ax[1].set_xticks(x); ax[1].set_xticklabels(tab.tipo)
ax[1].set_title('Conclusões por tipo (volume)'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()""")

md("""**Leitura:** das **33.631** conclusões de bogo/discount, apenas **70,4%
(23.677)** aconteceram *depois* de o cliente ver a oferta. As outras **29,6%
(9.954)** tiveram o desconto pago sem que a oferta tivesse influenciado a compra:
**17,0% nunca foram vistas** dentro da validade e **12,5% só foram vistas depois**
de o cupom já ter sido resgatado.

Por tipo, o padrão difere: **discount** tem mais conclusões no total (58,8% vs
51,4%), mas também mais resgates às cegas — **11,3%** das ofertas de discount são
completadas sem visualização, contra **7,5%** das bogo. Faz sentido pela mecânica:
discount tem gasto mínimo maior e prazo mais longo, então é mais fácil o cliente
atingir o valor no curso normal das compras, sem nunca abrir a oferta.

Esse é o desperdício que a estratégia precisa atacar — e a razão pela qual medir
apenas "quem completa" enviesaria a decisão de envio.""")

md("""# Caracterização descritiva dos segmentos de cliente

⚠️ **Escopo desta seção — e o que ela deliberadamente NÃO faz.**
Aqui apenas *descrevemos* como os clientes se distribuem. **Nada nesta seção é
causal**, e não usamos "completou oferta" como alvo de decisão. O motivo está
demonstrado logo abaixo: **completar uma oferta é, em boa medida, uma consequência
mecânica de gastar** — quem gasta muito cruza o valor mínimo de qualquer forma, com
ou sem influência do cupom.

Como o objetivo do negócio é **monetário e incremental** (vender mais *por causa*
da oferta), o eixo de interesse aqui é o **gasto**; e a pergunta causal — quanto do
gasto o envio realmente *causou* — só pode ser respondida com o grupo de controle
identificado no `1_data_processing.ipynb` e medido no `2_modeling.ipynb`.""")

md("""## Conclusão de oferta × volume de gasto
Teste da afirmação acima: a taxa de conclusão por quintil de gasto.""")
co("""trans = tx[tx.event=='transaction']
behav = trans.groupby('account_id').agg(
            n_transactions=('amount','size'),
            total_spend=('amount','sum'),
            avg_ticket=('amount','mean')).reset_index()
comp_cnt = (tx[tx.event=='offer completed'].groupby('account_id').size()
            .rename('n_completed').reset_index())

cust = profile.rename(columns={'id':'account_id'}).copy()
if not pd.api.types.is_datetime64_any_dtype(cust['registered_on']):
    cust['registered_on'] = pd.to_datetime(cust['registered_on'], format='%Y%m%d')
cust['incomplete_profile'] = (cust.age==118).astype(int)
cust['age'] = cust.age.replace(118, np.nan)
cust['account_age_years'] = ((cust.registered_on.max() - cust.registered_on).dt.days / 365.25).round(2)
cust['gender'] = cust.gender.fillna('unknown')
cust = cust.merge(behav, on='account_id', how='left').merge(comp_cnt, on='account_id', how='left')
cust[['n_transactions','total_spend','avg_ticket','n_completed']] = \\
    cust[['n_transactions','total_spend','avg_ticket','n_completed']].fillna(0)
cust['completou_alguma'] = (cust.n_completed>0).astype(int)

# conclusões "às cegas": completou sem ter visto antes (usa a atribuição já feita)
cegas = bd.groupby('account_id').apply(
    lambda g: (g.first_comp.notna() & (g.first_view.isna() | (g.first_view > g.first_comp))).sum(),
    include_groups=False).rename('n_cegas')
cust = cust.merge(cegas, on='account_id', how='left')
cust['n_cegas'] = cust.n_cegas.fillna(0)
print('clientes:', len(cust), '| taxa "completou alguma":', round(cust.completou_alguma.mean(),3))
cust.head()""")

co("""cust['q_gasto'] = pd.qcut(cust.total_spend.rank(method='first'), 5,
                          labels=['Q1 (baixo)','Q2','Q3','Q4','Q5 (alto)'])
taut = cust.groupby('q_gasto', observed=True).agg(
    gasto_medio=('total_spend','mean'),
    taxa_completou=('completou_alguma','mean'),
    conclusoes_cegas_por_cliente=('n_cegas','mean'))
print(taut.round(2).to_string())
print(f"\\ncorrelação gasto × completou_alguma : {cust.total_spend.corr(cust.completou_alguma):.3f}")
print(f"correlação gasto × conclusões CEGAS: {cust.total_spend.corr(cust.n_cegas):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar(taut.index.astype(str), taut.taxa_completou*100, color=IFOOD_RED)
for i, v in enumerate(taut.taxa_completou*100): ax[0].text(i, v, f'{v:.0f}%', ha='center', va='bottom')
ax[0].set_ylabel('% que completou alguma oferta'); ax[0].set_title('Alvo ingênuo é determinado pelo gasto')
ax[0].tick_params(rotation=0)
ax[1].bar(taut.index.astype(str), taut.conclusoes_cegas_por_cliente, color='#C62828')
ax[1].set_ylabel('conclusões às cegas por cliente')
ax[1].set_title('Quem gasta mais resgata mais SEM ver a oferta'); ax[1].tick_params(rotation=0)
plt.tight_layout(); plt.show()""")
md("""**Leitura:** a taxa de conclusão salta de **15%** no quintil de menor gasto
para **100%** no de maior — aritmética, não poder preditivo: quem gasta ~R$283 em 30
dias cruza um mínimo de R$10–20 em algum momento, tenha visto a oferta ou não.

O segundo gráfico mostra o outro lado: as **conclusões às cegas crescem junto com o
gasto** (0,04 → 1,09 por cliente). Os clientes que mais "completam ofertas" são
também os que mais resgatam cupom sem que a oferta tenha influenciado nada.""")

md("""## Perfil dos segmentos por **gasto**
Eixo de interesse: **quanto cada segmento gasta** — a variável alinhada ao objetivo
monetário.""")
co("""def perfil_por_gasto(df, var, bins=5, label=None):
    d = df.copy()
    if d[var].dtype.kind in 'if' and d[var].nunique() > bins:
        d = quantiliza_com_missing(d, var, var + '_b', bins)
        key = var + '_b'
    else:
        key = var
        d[key] = d[key].fillna('Missing').astype(str)
    g = d.groupby(key, observed=True).agg(
        n=('total_spend','size'), gasto_medio=('total_spend','mean'),
        ticket_medio=('avg_ticket','mean'), n_transacoes=('n_transactions','mean'))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(g.index.astype(str), g.gasto_medio, color='#4C72B0')
    ax.set_ylabel('gasto médio no período (R$)'); ax.set_title(label or f'Gasto médio por {var}')
    ax.tick_params(axis='x', rotation=0)
    for i, v in enumerate(g.gasto_medio): ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.show()
    return g.round(2)

display(perfil_por_gasto(cust, 'gender', label='Gasto médio por gênero'))
display(perfil_por_gasto(cust, 'incomplete_profile', label='Gasto médio: perfil completo vs incompleto'))""")
md("**Leitura:** perfis incompletos gastam consistentemente menos — são clientes "
   "menos engajados com a plataforma, não apenas com o cadastro. Diferenças por "
   "gênero no gasto são pequenas frente às comportamentais.")

md("### Idade, limite de crédito e idade da conta vs gasto")
co("""display(perfil_por_gasto(cust, 'age', label='Gasto médio por quintil de idade'))
display(perfil_por_gasto(cust, 'credit_card_limit', label='Gasto médio por quintil de limite'))
display(perfil_por_gasto(cust, 'account_age_years', label='Gasto médio por quintil de idade da conta'))""")
md("""**Leitura:** limite de crédito e idade crescem com o gasto — coerente com o
perfil socioeconômico já descrito (público maduro, renda estável). A idade da conta
também acompanha, sugerindo que o **relacionamento consolidado** vem junto com maior
volume de compra. *(Bucket -1 = valor ausente.)*""")

md("# Correlações entre variáveis numéricas (nível cliente)")
co("""num_cols = ['age','credit_card_limit','account_age_years','n_transactions',
            'total_spend','avg_ticket','n_completed','n_cegas']
high = print_correlation_matrix(cust, num_cols, corr_max=0.75)""")
md("""**Leitura:** o par mais forte é **`total_spend` × `avg_ticket` (0,78)** — único
acima do corte de 0,75, candidato natural a redução na feature selection. Já
`n_transactions` é praticamente ortogonal ao ticket (−0,06): volume e valor por
compra são dimensões independentes aqui, então vale manter as duas.""")

md("""# Identificação do grupo de controle
O item anterior mostrou que precisamos de um contrafactual. A cadência **em lote**
descoberta no overview de transações abre essa porta: se as ofertas saem em ondas,
alguém pode não ter recebido em uma dada onda — e esse alguém serve de comparação.""")
co("""ondas = sorted(rec.t_recv.unique())
todos = set(profile.account_id) if 'account_id' in profile.columns else set(profile.id)
linhas = []
for w in ondas:
    recebeu = set(rec.loc[rec.t_recv == w, 'account_id'])
    nao_recebeu = todos - recebeu
    # controle "limpo": além de não receber agora, não pode ter oferta de onda
    # anterior ainda vigente (senão está contaminado por tratamento passado)
    vigente = set(rec.loc[(rec.t_recv < w) & (rec.t_end > w), 'account_id'])
    linhas.append({'onda (dia)': int(w), 'recebeu': len(recebeu),
                   'não recebeu': len(nao_recebeu),
                   'controle limpo': len(nao_recebeu - vigente)})
ctrl = pd.DataFrame(linhas)
ctrl['% controle limpo'] = (ctrl['controle limpo'] / len(todos) * 100).round(1)
print(ctrl.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(ctrl['onda (dia)'].astype(str), ctrl['recebeu'], label='recebeu oferta', color='#4C72B0')
ax.bar(ctrl['onda (dia)'].astype(str), ctrl['controle limpo'],
       bottom=ctrl['recebeu'], label='controle limpo', color='#2E7D32')
ax.set_xlabel('onda (dia do teste)'); ax.set_ylabel('clientes')
ax.set_title('Cada onda deixa de fora ~25% da base — o grupo de controle')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()""")

md("""### O controle é comparável aos tratados? (balance check)
Ter um grupo sem oferta não basta: ele só serve como contrafactual se for
**parecido** com quem recebeu. Comparamos características medidas **antes** da onda
— se o envio foi essencialmente aleatório, as médias devem bater.""")
co("""W_CHECK = 17.0   # onda com histórico prévio suficiente para comparar
recebeu = set(rec.loc[rec.t_recv == W_CHECK, 'account_id'])
vigente = set(rec.loc[(rec.t_recv < W_CHECK) & (rec.t_end > W_CHECK), 'account_id'])
controle = todos - recebeu - vigente

trans_ev = tx[tx.event == 'transaction']
pre = (trans_ev[trans_ev.time_since_test_start < W_CHECK]
       .groupby('account_id').amount.agg(gasto_pre='sum', n_tx_pre='size'))
perfil_ix = profile.set_index('account_id' if 'account_id' in profile.columns else 'id')

def resumo(ids, nome):
    s = pre.reindex(list(ids)).fillna(0)
    p = perfil_ix.reindex(list(ids))
    return {'grupo': nome, 'n': len(ids),
            'gasto_pré': s.gasto_pre.mean(), 'transações_pré': s.n_tx_pre.mean(),
            'idade': p.age.replace(118, np.nan).mean(),
            '% perfil incompleto': (p.age == 118).mean()*100}

bal = pd.DataFrame([resumo(recebeu, 'recebeu (tratado)'), resumo(controle, 'controle limpo')])
print(bal.round(2).to_string(index=False))
print('\\nclientes que servem de controle limpo em ao menos uma onda:',
      f"{len(set().union(*[todos - set(rec.loc[rec.t_recv==w,'account_id']) - set(rec.loc[(rec.t_recv<w)&(rec.t_end>w),'account_id']) for w in ondas])):,}")""")
md("""**Leitura:** em cada onda, **~4,2–4,4 mil clientes (25%) não recebem oferta**.
Descontando quem ainda tem oferta anterior vigente, sobra um **controle limpo** que
vai de 4.350 (onda 0, a mais limpa por não haver passado) a ~1,2 mil nas ondas
finais — quando a cadência acelera e mais gente está sob oferta ativa.

O balance check é o que valida o desenho: tratados e controle têm **gasto prévio
(R$ 51,78 vs 47,57), número de transações (4,11 vs 3,77), idade (54,4 vs 55,0) e
taxa de perfil incompleto (12,9% vs 13,8%) praticamente iguais**. Nenhuma diferença
sugere seleção — o envio se comporta como **aleatorizado**, o que habilita comparar
os dois grupos diretamente.

É esse achado que torna o case tratável: sem ele, não haveria como separar a venda
que a oferta *causou* daquela que aconteceria de qualquer forma.""")

md("""# Conclusões da EDA (dados brutos)
1. **Qualidade:** `age==118` = perfil incompleto (~12,8%) ≡ nulos de gênero/limite →
   flag + imputação no processing. Sem concentração temporal: é falha estrutural de
   cadastro, não lote/migração pontual.
2. **Ofertas:** 4 BOGO, 4 discount, 2 informational; duração 3–10 dias define a
   **janela de atribuição** do processing; `informational` não tem "completar".
   Economia distinta: BOGO paga ~1x o gasto destravado, discount ~4x.
3. **Recompensa desperdiçada (achado central):** das 33.631 conclusões atribuídas a
   ofertas bogo/discount, só **70,4%** ocorreram após o cliente ver a oferta.
   **29,6%** dos cupons foram pagos sem influenciar a compra (17,0% nunca vistos +
   12,5% vistos só depois do resgate). Discount desperdiça mais que BOGO (11,3% vs
   7,5% das enviadas). *Nota: 33.631 atribuições vs 33.579 eventos brutos de
   `offer completed` — a diferença vem de ofertas repetidas com janelas sobrepostas,
   em que um mesmo resgate é atribuído a mais de um envio.*
4. **Conclusão de oferta acompanha o gasto** (15% no menor quintil → **100%** no
   maior), assim como as conclusões às cegas → o alvo de decisão precisa ser o
   **efeito incremental do envio**, o que exige grupo de controle.
5. **Perfil dos segmentos:** o **limite de crédito** é o que melhor acompanha o gasto
   (R$ 64 → R$ 182, monotônico). Idade e idade da conta sobem até o meio da
   distribuição e depois **estabilizam ou caem** — relação não linear. Entre as
   comportamentais, só `total_spend` × `avg_ticket` passa de 0,75 de correlação;
   `n_transactions` é ortogonal ao ticket (−0,06).
6. **Timing:** envios são **em lote** em 6 dias (0,7,14,17,21,24), com cadência que
   **acelera** (de semanal para cada 3–4 dias); cada envio gera uma onda decrescente
   de visualização/conclusão — relevante para timing de campanha.
7. **Existe grupo de controle utilizável (viabiliza todo o resto):** cada onda deixa
   **~25% da base sem oferta**; descontando quem tem oferta anterior vigente, o
   controle limpo vai de 4.350 (onda 0) a ~1,2 mil (ondas finais). O balance check na
   onda 17 mostra tratados e controle equivalentes em gasto prévio, transações, idade
   e perfil incompleto → **o envio se comporta como aleatorizado**.

→ Próximo passo: `1_data_processing.ipynb` — formalizar o grupo de controle do item 7
em uma tabela (cliente × onda) com features estritamente pré-onda, para medir no
`2_modeling.ipynb` o efeito incremental exigido pelo item 4.
""")

nb["cells"] = c
out = Path(__file__).resolve().parents[1] / "notebooks" / "0_eda.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
