"""Templates de prompt pra explicacao em linguagem natural do resultado do modelo.
DISCLAIMER e a fonte unica desse texto — api/main.py importa daqui em vez de duplicar
(evita import circular: llm nunca importa de api)."""

DISCLAIMER = (
    "Ferramenta de apoio a decisao. Nao substitui avaliacao medica, mamografia, "
    "ultrassonografia nem biopsia. O profissional de saude tem a palavra final."
)

SYSTEM_PROMPT = f"""Voce e um assistente que traduz a saida de um modelo de machine \
learning de classificacao benigno/maligno (cancer de mama, dataset Wisconsin \
Diagnostic) em uma explicacao clara para um profissional de saude, sem jargao \
estatistico desnecessario.

Regras obrigatorias:
- Nunca invente numeros: use apenas os valores fornecidos no prompt do usuario.
- Explique o resultado (label + probabilidades) e cite 2-3 das features mais \
relevantes fornecidas, sem repetir a lista inteira.
- Termine sempre com este aviso, literalmente: "{DISCLAIMER}"
- Responda em portugues, em no maximo 3 paragrafos curtos.
"""

USER_PROMPT_TEMPLATE = """Resultado do modelo:
- Predicao: {label} ({predicao})
- Probabilidade benigno: {probabilidade_benigno:.1%}
- Probabilidade maligno: {probabilidade_maligno:.1%}

Medidas mais relevantes da amostra (nucleo celular, PAAF):
{features_relevantes}

Gere a explicacao para o profissional de saude, seguindo as regras do system prompt."""


def build_user_prompt(prediction: dict, relevant_features: dict[str, float]) -> str:
    features_lines = "\n".join(
        f"- {name}: {value:.4g}" for name, value in relevant_features.items()
    )
    return USER_PROMPT_TEMPLATE.format(
        label=prediction["label"],
        predicao=prediction["predicao"],
        probabilidade_benigno=prediction["probabilidade_benigno"],
        probabilidade_maligno=prediction["probabilidade_maligno"],
        features_relevantes=features_lines,
    )
