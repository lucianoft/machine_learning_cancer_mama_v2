"""CLI: roda o algoritmo genetico em >=3 configuracoes diferentes (populacao/mutacao),
pra logistic_regression e random_forest, plota a convergencia de cada experimento,
recalcula o baseline (hiperparametros originais do notebook da Fase 1) no mesmo
treino/teste, compara contra o melhor individuo encontrado pelo GA, e salva a pipeline
otimizada vencedora em models/.

Uso: python -m ga.run_ga_experiments
"""

import logging
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ga.data import RANDOM_STATE, build_preprocessor, load_train_test  # noqa: E402
from ga.genetic_algorithm import GAResult, run_ga  # noqa: E402
from ga.individual import build_estimator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

MODEL_FAMILIES = ["logistic_regression", "random_forest"]

# >=3 experimentos com configuracoes diferentes de GA, por requisito do PDF (tamanho de
# populacao, taxa de mutacao). cv=3 e pop/gens moderados pra manter o tempo de execucao
# do script razoavel — random_forest com muitas arvores por individuo e o gargalo.
EXPERIMENTS = [
    {"name": "baseline_ga", "population_size": 16, "n_generations": 10, "mutation_rate": 0.10, "crossover_rate": 0.8},
    {"name": "populacao_maior", "population_size": 32, "n_generations": 10, "mutation_rate": 0.10, "crossover_rate": 0.8},
    {"name": "mutacao_alta", "population_size": 16, "n_generations": 10, "mutation_rate": 0.35, "crossover_rate": 0.8},
]
CV_FOLDS = 3


def original_estimator(model_family: str):
    """Hiperparametros exatamente como no notebook da Fase 1 (cell 27) — baseline
    recalculado no mesmo treino/teste pra comparacao justa com o GA."""
    if model_family == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    if model_family == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE)
    raise ValueError(model_family)


def evaluate_on_test(pipeline: Pipeline, x_test, y_test) -> dict:
    y_pred = pipeline.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def plot_convergence(result: GAResult, out_path: Path) -> None:
    generations = [h.generation for h in result.history]
    best = [h.best_fitness for h in result.history]
    avg = [h.avg_fitness for h in result.history]

    cfg = result.config
    subtitle = (
        f"pop={cfg['population_size']}  mutacao={cfg['mutation_rate']}  "
        f"crossover={cfg['crossover_rate']}  geracoes={cfg['n_generations']}"
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(generations, best, label="Melhor fitness", marker="o")
    plt.plot(generations, avg, label="Fitness medio", marker="x", linestyle="--")
    plt.xlabel("Geracao")
    plt.ylabel("Fitness (0.25*accuracy + 0.45*recall + 0.30*f1)")
    plt.suptitle(f"Convergencia GA — {result.model_family}", fontsize=12, y=0.98)
    plt.title(subtitle, fontsize=9, color="#555555")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    x_train, x_test, y_train, y_test, feature_cols, _ = load_train_test()

    all_results: list[dict] = []
    best_by_family: dict[str, tuple[GAResult, dict]] = {}

    for model_family in MODEL_FAMILIES:
        best_test_f1 = -1.0
        for exp in EXPERIMENTS:
            logger.info("=== %s | experimento=%s ===", model_family, exp["name"])
            result = run_ga(
                model_family=model_family,
                x_train=x_train,
                y_train=y_train,
                feature_cols=feature_cols,
                population_size=exp["population_size"],
                n_generations=exp["n_generations"],
                mutation_rate=exp["mutation_rate"],
                crossover_rate=exp["crossover_rate"],
                cv=CV_FOLDS,
                random_state=RANDOM_STATE,
            )

            plot_path = REPORTS_DIR / f"ga_convergence_{model_family}_{exp['name']}.png"
            plot_convergence(result, plot_path)

            # reajusta o melhor individuo do experimento no treino completo e avalia no teste
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                ("classifier", build_estimator(model_family, result.best_individual)),
            ])
            pipeline.fit(x_train, y_train)
            test_metrics = evaluate_on_test(pipeline, x_test, y_test)

            all_results.append({
                "model_family": model_family,
                "experimento": exp["name"],
                "population_size": exp["population_size"],
                "n_generations": exp["n_generations"],
                "mutation_rate": exp["mutation_rate"],
                "cv_fitness": result.best_fitness_result.fitness,
                "cv_accuracy": result.best_fitness_result.accuracy,
                "cv_recall": result.best_fitness_result.recall,
                "cv_f1": result.best_fitness_result.f1,
                "test_accuracy": test_metrics["accuracy"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "hiperparametros": result.best_individual,
            })

            if test_metrics["f1"] > best_test_f1:
                best_test_f1 = test_metrics["f1"]
                best_by_family[model_family] = (result, test_metrics)

    results_df = pd.DataFrame(all_results)
    results_csv_path = REPORTS_DIR / "ga_experiments_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info("Resultados dos experimentos salvos em %s", results_csv_path)

    # baseline: hiperparametros originais do notebook, no mesmo treino/teste
    comparison_rows = []
    for model_family in MODEL_FAMILIES:
        baseline_pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor(feature_cols)),
            ("classifier", original_estimator(model_family)),
        ])
        baseline_pipeline.fit(x_train, y_train)
        baseline_metrics = evaluate_on_test(baseline_pipeline, x_test, y_test)

        ga_result, ga_test_metrics = best_by_family[model_family]
        comparison_rows.append({
            "model_family": model_family,
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_recall": baseline_metrics["recall"],
            "baseline_f1": baseline_metrics["f1"],
            "ga_accuracy": ga_test_metrics["accuracy"],
            "ga_recall": ga_test_metrics["recall"],
            "ga_f1": ga_test_metrics["f1"],
            "ga_hiperparametros": ga_result.best_individual,
            "ga_config": ga_result.config,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = REPORTS_DIR / "ga_vs_baseline.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info("Comparativo otimizado vs. original salvo em %s", comparison_csv_path)
    print(comparison_df.to_string(index=False))

    # modelo vencedor geral (maior F1 no teste entre as duas familias) vira a pipeline
    # servida pela API
    winner_family = max(MODEL_FAMILIES, key=lambda f: best_by_family[f][1]["f1"])
    winner_result, winner_metrics = best_by_family[winner_family]

    winner_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(feature_cols)),
        ("classifier", build_estimator(winner_family, winner_result.best_individual)),
    ])
    winner_pipeline.fit(x_train, y_train)

    model_path = MODELS_DIR / "pipeline_breast_cancer_ga_optimized.pkl"
    metadata_path = MODELS_DIR / "pipeline_metadata_ga_optimized.pkl"

    joblib.dump(winner_pipeline, model_path)
    joblib.dump({
        "model_name": f"{winner_family} (otimizado via algoritmo genetico)",
        "model_family": winner_family,
        "feature_columns": feature_cols,
        "target_column": "diagnosis",
        "target_labels": {0: "Benigno", 1: "Maligno"},
        "dataset": "breast_cancer_wisconsin.csv",
        "ga_hiperparametros": winner_result.best_individual,
        "ga_config": winner_result.config,
        "test_metrics": winner_metrics,
    }, metadata_path)

    logger.info("Vencedor: %s | F1 teste=%.4f | salvo em %s", winner_family, winner_metrics["f1"], model_path)


if __name__ == "__main__":
    main()
