"""Carga e split dos dados — replica exatamente as seções 1, 3 e 4 do notebook
(Tech_Challenge_Breast_Cancer.ipynb), pra que o GA treine/avalie sobre o mesmo
treino/teste usados na Fase 1. Nao duplica a logica em cada script: quem precisa dos
dados (fitness, run_ga_experiments, testes) importa daqui.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "diagnosis"

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "breast_cancer_wisconsin.csv"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].map({"B": 0, "M": 1})
    return df


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            ),
        ]
    )


def load_train_test():
    """Retorna (X_train, X_test, y_train, y_test, feature_cols, preprocessor) —
    mesmo split 80/20 estratificado, random_state=42, do notebook."""
    df = load_dataset()
    feature_cols = [col for col in df.columns if col not in ["id", TARGET]]

    x = df[feature_cols]
    y = df[TARGET].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(feature_cols)
    return x_train, x_test, y_train, y_test, feature_cols, preprocessor
