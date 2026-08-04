"""Representacao do individuo: um dict {nome_do_gene: valor}, decodificado direto pros
kwargs do classificador sklearn. Nao usamos bitstring — pra um espaco misto
int/float/categorico, um dict tipado e uma codificacao mais direta e menos propensa a
erro de decodificacao do que empacotar tudo em binario.
"""

import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ga.data import RANDOM_STATE
from ga.search_space import SEARCH_SPACES, SOLVER_BY_PENALTY

Individual = dict[str, object]


def random_gene(gene_spec: tuple, rng: random.Random):
    kind = gene_spec[0]
    if kind == "log_float":
        _, low, high = gene_spec
        return round(10 ** rng.uniform(_log10(low), _log10(high)), 6)
    if kind == "int":
        _, low, high = gene_spec
        return rng.randint(low, high)
    if kind == "categorical":
        _, choices = gene_spec
        return rng.choice(choices)
    raise ValueError(f"Tipo de gene desconhecido: {kind}")


def _log10(value: float) -> float:
    import math
    return math.log10(value)


def random_individual(model_family: str, rng: random.Random) -> Individual:
    space = SEARCH_SPACES[model_family]
    return {gene: random_gene(spec, rng) for gene, spec in space.items()}


def clip_gene(gene_spec: tuple, value):
    kind = gene_spec[0]
    if kind in ("log_float", "int"):
        _, low, high = gene_spec
        value = max(low, min(high, value))
        return int(round(value)) if kind == "int" else float(value)
    return value


def build_estimator(model_family: str, individual: Individual):
    """Decodifica o individuo pro classificador sklearn correspondente."""
    if model_family == "logistic_regression":
        penalty = individual["penalty"]
        return LogisticRegression(
            C=individual["C"],
            penalty=penalty,
            solver=SOLVER_BY_PENALTY[penalty],
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=individual["n_estimators"],
            max_depth=individual["max_depth"],
            min_samples_split=individual["min_samples_split"],
            min_samples_leaf=individual["min_samples_leaf"],
            max_features=individual["max_features"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Familia de modelo desconhecida: {model_family}")
