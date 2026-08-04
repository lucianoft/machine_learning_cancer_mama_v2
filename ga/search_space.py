"""Espaco de busca (genes) por familia de modelo. Cada gene e uma tupla
(tipo, ...limites), onde tipo e um dos:
  - "log_float": (min, max) amostrado em escala log — bom pra parametros como C
  - "int": (min, max) inteiro
  - "categorical": lista de valores possiveis

Random Forest e o modelo com o espaco de busca mais rico (mais graus de liberdade
pro GA mostrar ganho); Logistic Regression e o modelo hoje em producao (Fase 1),
entao otimiza-lo da uma comparacao direta "antes vs depois" do mesmo modelo servido.
"""

SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "logistic_regression": {
        "C": ("log_float", 0.001, 100.0),
        "penalty": ("categorical", ["l1", "l2"]),
    },
    "random_forest": {
        "n_estimators": ("int", 30, 150),
        "max_depth": ("int", 2, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
}

# penalty=l1 so e suportado pelo solver 'liblinear' (e 'saga', mais lento); l2 aceita
# ambos. Como o gene "penalty" e escolhido pelo GA, o solver tem que ser derivado dele
# em tempo de decode, nao sorteado independente (evita combinacao invalida C/penalty/solver).
SOLVER_BY_PENALTY = {"l1": "liblinear", "l2": "liblinear"}
