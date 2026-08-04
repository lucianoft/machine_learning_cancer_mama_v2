from ga.data import load_train_test
from ga.fitness import evaluate


def test_evaluate_retorna_metricas_no_intervalo_valido():
    x_train, _, y_train, _, feature_cols, _ = load_train_test()
    individual = {"C": 1.0, "penalty": "l2"}

    result = evaluate(individual, "logistic_regression", x_train, y_train, feature_cols, cv=3)

    assert 0.0 <= result.fitness <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_evaluate_e_reprodutivel_com_mesmo_individuo():
    x_train, _, y_train, _, feature_cols, _ = load_train_test()
    individual = {"C": 0.5, "penalty": "l2"}

    result_a = evaluate(individual, "logistic_regression", x_train, y_train, feature_cols, cv=3)
    result_b = evaluate(individual, "logistic_regression", x_train, y_train, feature_cols, cv=3)

    assert result_a.fitness == result_b.fitness


def test_evaluate_random_forest():
    x_train, _, y_train, _, feature_cols, _ = load_train_test()
    individual = {
        "n_estimators": 40,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    result = evaluate(individual, "random_forest", x_train, y_train, feature_cols, cv=3)

    assert result.fitness > 0.0
