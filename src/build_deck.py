"""Generates the 5-slide stakeholder deck (presentation/ifood_case.pptx)."""
from pathlib import Path
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
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

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


def title_slide(slide, kicker, title, subtitle):
    band(slide)
    add_text(slide, 0.9, 2.2, 11.5, 1.0, kicker, 18, RED, bold=True)
    add_text(slide, 0.9, 2.9, 11.5, 1.6, title, 40, DARK, bold=True)
    add_text(slide, 0.9, 4.6, 11.5, 1.2, subtitle, 18, GREY)


def content_slide(kicker, title):
    s = prs.slides.add_slide(BLANK)
    band(s)
    add_text(s, 0.7, 0.45, 12, 0.5, kicker, 14, RED, bold=True)
    add_text(s, 0.7, 0.85, 12, 0.9, title, 28, DARK, bold=True)
    return s


# ---------------------------------------------------------------- Slide 1 (capa)
s = prs.slides.add_slide(BLANK)
title_slide(
    s,
    "iFood · Data Science — Otimização de Cupons",
    "Enviar a oferta certa, para o cliente certo",
    "Uplift modeling para maximizar o retorno incremental dos cupons — "
    "parando de gastar recompensa com quem já compraria.",
)
add_text(s, 0.9, 6.4, 11.5, 0.6,
         "Recomendação de oferta por cliente  ·  baseada em ~306 mil eventos e 17 mil clientes",
         13, GREY)

# ---------------------------------------------------------- Slide 2 (o problema)
s = content_slide("O problema", "Hoje parte do orçamento de cupons é desperdiçada")
s.shapes.add_picture(str(FIG / "01_funnel.png"), Inches(0.6), Inches(2.0), width=Inches(7.3))
add_text(s, 8.2, 2.1, 4.6, 4.5,
         "1 em cada 10 cupons\né completado SEM ser visto",
         22, RED, bold=True)
add_text(s, 8.2, 3.2, 4.7, 4.0,
         "→ O cliente compraria de qualquer forma.\n"
         "   A recompensa é dinheiro jogado fora.\n\n"
         "Além disso, 31% veem o cupom e não\nusam — mídia gasta sem retorno.\n\n"
         "Enviar para todo mundo não é a\nestratégia mais rentável.",
         15, DARK)

# ---------------------------------------------------- Slide 3 (a solução/insight)
s = content_slide("A solução", "Prever o efeito INCREMENTAL de cada oferta (uplift)")
add_text(s, 0.7, 1.8, 6.0, 3.4,
         "Em vez de prever quem completa,\n"
         "prevemos quanto a oferta MUDA o\ncomportamento de cada cliente.\n\n"
         "Isso separa 3 grupos:\n\n"
         "•  Persuadíveis → ENVIAR\n"
         "•  Já comprariam → NÃO enviar (economia)\n"
         "•  Não respondem → NÃO enviar",
         16, DARK)
add_text(s, 0.7, 5.4, 6.2, 1.5,
         "Descoberta: cupons de DESCONTO têm uplift\n"
         "~4x maior que BOGO (+29 vs +8 p.p.).",
         15, RED, bold=True)
s.shapes.add_picture(str(FIG / "02_uplift_segments.png"), Inches(6.9), Inches(2.2), width=Inches(6.1))

# ----------------------------------------------------- Slide 4 (impacto/projeção)
s = content_slide("O impacto", "Direcionar por uplift entrega mais valor com menos envios")
s.shapes.add_picture(str(FIG / "04_business_impact.png"), Inches(0.5), Inches(1.9), width=Inches(7.2))
add_text(s, 7.9, 2.0, 5.0, 0.6, "Projeção por 1 milhão de cupons*", 15, GREY, bold=True)
for i, (big, small) in enumerate([
    ("+R$ 1,45M", "de valor líquido incremental (+21%)\nvs enviar para todos"),
    ("−20%", "de cupons enviados (menos custo\nde mídia e de recompensa)"),
    ("+13%", "mais conclusões incrementais,\nao cortar quem tem uplift negativo"),
]):
    y = 2.7 + i * 1.35
    add_text(s, 7.9, y, 5.0, 0.6, big, 26, RED, bold=True)
    add_text(s, 7.9, y + 0.55, 5.0, 0.8, small, 13, DARK)
add_text(s, 0.6, 6.95, 12, 0.4,
         "*Escala das taxas observadas em holdout; ticket médio/conclusão R$52,60, recompensa média R$4,90.",
         10, GREY)

# --------------------------------------------------- Slide 5 (como / próximos)
s = content_slide("Como funciona e próximos passos", "Pronto para um piloto controlado")
add_text(s, 0.7, 1.9, 6.0, 4.5,
         "Como funciona\n",
         18, DARK, bold=True)
add_text(s, 0.7, 2.6, 6.1, 4.5,
         "1.  Modelo de uplift (T-learner/XGBoost)\n     pontua cada cliente × oferta.\n\n"
         "2.  Para cada cliente, recomenda a oferta\n     de maior uplift — ou nenhuma.\n\n"
         "3.  Só envia a quem tem retorno incremental\n     positivo (≈10% da base não deve receber).\n\n"
         "Qualidade do modelo: ranqueia o uplift\n~2x melhor que a abordagem padrão (Qini).",
         15, DARK)
add_text(s, 7.2, 1.9, 5.6, 0.7, "Próximos passos", 18, DARK, bold=True)
add_text(s, 7.2, 2.6, 5.7, 4.5,
         "•  A/B test com grupo de controle real\n   (holdout sem oferta) para validar o uplift.\n\n"
         "•  Otimização de timing e canal do envio.\n\n"
         "•  Alocação sob orçamento (uplift por R$\n   investido).\n\n"
         "•  Colocar o score em produção no motor\n   de campanhas de CRM.",
         15, DARK)

prs.save(str(OUT))
print("wrote", OUT)
