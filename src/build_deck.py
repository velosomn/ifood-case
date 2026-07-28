"""Generates the 5-slide stakeholder deck (presentation/ifood_case.pptx).

Narrativa: desperdício de cupom -> efeito causal medido -> política validada ->
impacto (piso model-free + teto a validar em A/B).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "presentation" / "figures"
OUT = ROOT / "presentation" / "ifood_case.pptx"

RED = RGBColor(0xEA, 0x1D, 0x2C)
DARK = RGBColor(0x2B, 0x2B, 0x2B)
GREY = RGBColor(0x66, 0x66, 0x66)

# ------------------------------------------------ figura do funil (nºs c/ ordem)
plt.rcParams["figure.dpi"] = 110
seg = [("Viu e usou o cupom", 38.8, "#2E7D32"),
       ("Viu e não usou", 31.2, "#F9A825"),
       ("Nem viu, nem usou", 13.7, "#9E9E9E"),
       ("Usou SEM ter visto", 9.4, "#EA1D2C"),
       ("Usou e viu só depois", 6.9, "#C62828")]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
labels = [s[0] for s in seg][::-1]
vals = [s[1] for s in seg][::-1]
cols = [s[2] for s in seg][::-1]
ax.barh(labels, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=10)
ax.set_xlabel("% das ofertas enviadas (bogo/discount)")
ax.set_title("O que acontece com cada oferta enviada")
ax.set_xlim(0, 46)
plt.tight_layout()
plt.savefig(FIG / "10_funnel.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ helpers pptx
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_text(slide, left, top, width, height, text, size, color=DARK, bold=False,
             align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for line in text.split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return tb


def band(slide):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(0.25))
    bar.fill.solid(); bar.fill.fore_color.rgb = RED; bar.line.fill.background()


def content_slide(kicker, title):
    s = prs.slides.add_slide(BLANK)
    band(s)
    add_text(s, 0.7, 0.45, 12, 0.5, kicker, 14, RED, bold=True)
    add_text(s, 0.7, 0.85, 12.2, 0.9, title, 27, DARK, bold=True)
    return s


# ------------------------------------------------------------------ Slide 1 capa
s = prs.slides.add_slide(BLANK)
band(s)
add_text(s, 0.9, 2.2, 11.5, 1.0, "iFood · Data Science — Direcionamento de Cupons", 18, RED, bold=True)
add_text(s, 0.9, 2.9, 11.8, 1.6, "A oferta certa, para o cliente certo —\ne só quando ela gera venda nova", 38, DARK, bold=True)
add_text(s, 0.9, 4.9, 11.5, 1.2,
         "Medimos o efeito real de cada envio (venda incremental, descontado o cupom)\n"
         "e transformamos isso numa política de envio por cliente.", 17, GREY)
add_text(s, 0.9, 6.5, 11.5, 0.5,
         "Base: 17 mil clientes · 306 mil eventos · 76 mil envios em 6 campanhas", 12, GREY)

# ------------------------------------------------------------- Slide 2 problema
s = content_slide("O problema", "Parte do orçamento de cupons não gera venda nova")
s.shapes.add_picture(str(FIG / "10_funnel.png"), Inches(0.6), Inches(2.0), width=Inches(7.2))
add_text(s, 8.1, 2.1, 4.8, 0.9, "~30% dos cupons pagos\nnão influenciaram a compra", 21, RED, bold=True)
add_text(s, 8.1, 3.5, 4.9, 3.2,
         "O cliente atingiu o valor mínimo e o\ncupom foi resgatado sem que ele\n"
         "tivesse visto a oferta — a compra\naconteceria de qualquer forma.\n\n"
         "Enviar a mesma oferta para todos\nignora quem responde e quem não.", 15, DARK)

# ------------------------------------------------------- Slide 3 efeito causal
s = content_slide("Quanto vale um envio", "Cada envio gera, em média, R$ 9 de venda nova — já descontado o cupom")
s.shapes.add_picture(str(FIG / "05_ate.png"), Inches(0.6), Inches(2.0), width=Inches(6.8))
add_text(s, 7.7, 2.1, 5.2, 0.8, "Como medimos", 16, DARK, bold=True)
add_text(s, 7.7, 2.7, 5.3, 3.6,
         "Em cada campanha, ~25% dos clientes\nnão receberam oferta — um grupo de\n"
         "comparação equivalente aos demais.\n\n"
         "A diferença entre os dois grupos é a\nvenda que o envio causou:\n"
         "R$ 9,21 por cliente em 7 dias\n(intervalo de confiança 8,45–9,90).\n\n"
         "Cupons de desconto rendem mais que\nBOGO e que ofertas informativas.", 14, DARK)

# ---------------------------------------------------------- Slide 4 política
s = content_slide("A solução", "Uma política por cliente: a melhor oferta — ou nenhuma")
s.shapes.add_picture(str(FIG / "08_policy_validation.png"), Inches(0.6), Inches(2.0), width=Inches(6.8))
add_text(s, 7.7, 2.0, 5.2, 0.8, "O modelo prioriza quem gera valor", 16, DARK, bold=True)
add_text(s, 7.7, 2.6, 5.3, 2.6,
         "Testado em campanhas futuras (fora da\namostra de treino): o grupo apontado\n"
         "como prioritário gerou R$ 21 de venda\nincremental por cliente; o grupo de\n"
         "menor prioridade, zero.", 14, DARK)
add_text(s, 7.7, 4.7, 5.2, 1.6,
         "Hoje, só 10% dos clientes\nrecebem a sua melhor oferta.", 19, RED, bold=True)

# ------------------------------------------------------------ Slide 5 impacto
s = content_slide("Impacto e próximos passos", "De R$ 10,3M para R$ 12,3M por milhão de envios — com upside a testar")
s.shapes.add_picture(str(FIG / "09_business_impact.png"), Inches(0.5), Inches(2.0), width=Inches(6.9))
add_text(s, 7.7, 2.0, 5.3, 1.9,
         "+R$ 2,0M (+20%) por 1M de envios já\nno cenário conservador, enviando 5%\nmenos cupons.\n"
         "Potencial de ~R$ 29M com persona-\nlização total — número de modelo, a\nconfirmar em teste controlado.", 14, DARK)
add_text(s, 7.7, 4.1, 5.2, 0.6, "Próximos passos", 16, DARK, bold=True)
add_text(s, 7.7, 4.7, 5.3, 2.2,
         "1. Teste A/B com grupo de controle\n    para confirmar os ganhos\n"
         "2. Alocação sob orçamento de cupons\n"
         "3. Otimização de timing e canal\n"
         "4. Integração ao motor de campanhas", 14, DARK)

prs.save(str(OUT))
print("wrote", OUT)
