"""Funcao fitness: decodifica o individuo num classificador, encaixa no mesmo
pre-processamento do notebook (imputer + scaler) e avalia por cross-validation
*no treino* — nunca no teste, senao o GA overfita no conjunto que devia ficar de fora
pra comparacao final. Combina accuracy/recall/F1 num escalar, com recall pesando mais:
falso negativo (maligno classificado como benigno) e o erro mais grave nesse contexto
clinico, como ja registrado no RELATORIO_TECNICO.md da Fase 1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ga.data import RANDOM_STATE, build_preprocessor
from ga.individual import Individual, build_estimator

FITNESS_WEIGHTS = {"accuracy": 0.25, "recall": 0.45, "f1": 0.30}


@dataclass
class FitnessResult:
    fitness: float
    accuracy: float
    recall: float
    f1: float


def evaluate(
    individual: Individual,
    model_family: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    cv: int = 5,
) -> FitnessResult:
    estimator = build_estimator(model_family, individual)
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(feature_cols)),
        ("classifier", estimator),
    ])

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_validate(
            pipeline, x_train, y_train,
            cv=splitter,
            scoring=["accuracy", "recall", "f1"],
            n_jobs=None,
            error_score="raise",
        )
    except ValueError:
        # combinacao de hiperparametros invalida pro sklearn (ex.: min_samples_leaf
        # > amostras da folha) — penaliza em vez de derrubar o experimento inteiro
        return FitnessResult(fitness=0.0, accuracy=0.0, recall=0.0, f1=0.0)

    accuracy = float(np.mean(scores["test_accuracy"]))
    recall = float(np.mean(scores["test_recall"]))
    f1 = float(np.mean(scores["test_f1"]))

    fitness = (
        FITNESS_WEIGHTS["accuracy"] * accuracy
        + FITNESS_WEIGHTS["recall"] * recall
        + FITNESS_WEIGHTS["f1"] * f1
    )
    return FitnessResult(fitness=fitness, accuracy=accuracy, recall=recall, f1=f1)
