"""Roda a explicacao da LLM em casos representativos (benigno/maligno, confianca
alta/baixa) e monta uma tabela pra avaliacao de qualidade — parte automatica (contem o
disclaimer? o texto nao veio vazio?) e colunas em branco pra revisao manual (clareza,
correcao clinica), preenchidas na hora de escrever o RELATORIO_TECNICO.md.

Uso: python -m llm.evaluation   (precisa de OPENAI_API_KEY no ambiente)
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

from ga.data import load_train_test
from llm.explainer import LLMUnavailableError, gerar_explicacao
from llm.prompts import DISCLAIMER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

N_CASOS = 8


def pick_representative_cases(x_test, y_test, pipeline, feature_cols, n: int = N_CASOS):
    """Mistura casos benignos/malignos com confianca alta e baixa, pra cobrir o espectro
    de explicacoes que a LLM vai precisar gerar."""
    proba = pipeline.predict_proba(x_test)
    confidence = proba.max(axis=1)

    df = x_test.copy()
    df["_y_real"] = y_test.values
    df["_confidence"] = confidence
    df = df.sort_values("_confidence")

    baixa_confianca = df.head(n // 2)
    alta_confianca = df.tail(n - n // 2)
    return pd.concat([baixa_confianca, alta_confianca])


def main() -> None:
    model_path = MODELS_DIR / "pipeline_breast_cancer_ga_optimized.pkl"
    metadata_path = MODELS_DIR / "pipeline_metadata_ga_optimized.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} nao encontrado — rode `python -m ga.run_ga_experiments` primeiro."
        )

    pipeline = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    feature_cols = metadata["feature_columns"]
    labels = metadata["target_labels"]

    _, x_test, _, y_test, _, _ = load_train_test()
    casos = pick_representative_cases(x_test, y_test, pipeline, feature_cols)

    rows = []
    for idx, row in casos.iterrows():
        patient_features = {col: row[col] for col in feature_cols}
        amostra = pd.DataFrame([patient_features])[feature_cols]
        pred = pipeline.predict(amostra)[0]
        proba = pipeline.predict_proba(amostra)[0]
        prediction = {
            "predicao": int(pred),
            "label": labels[int(pred)],
            "probabilidade_benigno": float(proba[0]),
            "probabilidade_maligno": float(proba[1]),
        }

        try:
            explicacao = gerar_explicacao(prediction, patient_features, pipeline, feature_cols)
        except LLMUnavailableError as exc:
            logger.warning("Caso %s: LLM indisponivel (%s)", idx, exc)
            explicacao = None

        rows.append({
            "caso_idx": idx,
            "label_real": labels[int(row["_y_real"])],
            "label_predito": prediction["label"],
            "confianca": max(proba),
            "explicacao": explicacao,
            "contem_disclaimer": bool(explicacao and DISCLAIMER in explicacao),
            "explicacao_nao_vazia": bool(explicacao and len(explicacao.strip()) > 0),
            "clareza_1_a_5": None,       # preencher manualmente na revisao
            "correcao_clinica_1_a_5": None,  # preencher manualmente na revisao
        })

    result_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "llm_evaluation.csv"
    result_df.to_csv(out_path, index=False)
    logger.info("Avaliacao da LLM salva em %s (%d casos)", out_path, len(result_df))
    print(result_df[["caso_idx", "label_real", "label_predito", "confianca", "contem_disclaimer"]].to_string(index=False))


if __name__ == "__main__":
    main()
