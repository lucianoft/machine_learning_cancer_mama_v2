"""Loop principal do algoritmo genetico: inicializa populacao, avalia fitness,
aplica elitismo + selecao por torneio + crossover + mutacao a cada geracao, e guarda
o historico (melhor/media por geracao) pra plotar a curva de convergencia.
"""

import logging
import random
from dataclasses import dataclass, field

import pandas as pd

from ga.fitness import FitnessResult, evaluate
from ga.individual import Individual, random_individual
from ga.operators import crossover, mutate, tournament_selection

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    avg_fitness: float
    best_individual: Individual


@dataclass
class GAResult:
    model_family: str
    config: dict
    best_individual: Individual
    best_fitness_result: FitnessResult
    history: list[GenerationStats] = field(default_factory=list)


def run_ga(
    model_family: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    population_size: int = 20,
    n_generations: int = 20,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism: int = 2,
    tournament_k: int = 3,
    cv: int = 5,
    random_state: int = 42,
) -> GAResult:
    rng = random.Random(random_state)
    config = {
        "population_size": population_size,
        "n_generations": n_generations,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "elitism": elitism,
        "tournament_k": tournament_k,
        "cv": cv,
    }

    population = [random_individual(model_family, rng) for _ in range(population_size)]
    history: list[GenerationStats] = []
    best_individual: Individual | None = None
    best_result: FitnessResult | None = None

    for generation in range(n_generations):
        results = [
            evaluate(ind, model_family, x_train, y_train, feature_cols, cv=cv)
            for ind in population
        ]
        fitnesses = [r.fitness for r in results]

        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fitness = fitnesses[gen_best_idx]
        if best_result is None or gen_best_fitness > best_result.fitness:
            best_individual = dict(population[gen_best_idx])
            best_result = results[gen_best_idx]

        history.append(GenerationStats(
            generation=generation,
            best_fitness=gen_best_fitness,
            avg_fitness=sum(fitnesses) / len(fitnesses),
            best_individual=dict(population[gen_best_idx]),
        ))
        logger.info(
            "geracao=%d melhor_fitness=%.4f media_fitness=%.4f",
            generation, gen_best_fitness, history[-1].avg_fitness,
        )

        # elitismo: os N melhores da geracao passam direto, sem crossover/mutacao
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        next_population = [dict(population[i]) for i in ranked[:elitism]]

        while len(next_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, rng, k=tournament_k)
            parent_b = tournament_selection(population, fitnesses, rng, k=tournament_k)
            child_a, child_b = crossover(parent_a, parent_b, model_family, rng, crossover_rate)
            child_a = mutate(child_a, model_family, rng, mutation_rate)
            child_b = mutate(child_b, model_family, rng, mutation_rate)
            next_population.append(child_a)
            if len(next_population) < population_size:
                next_population.append(child_b)

        population = next_population

    assert best_individual is not None and best_result is not None
    return GAResult(
        model_family=model_family,
        config=config,
        best_individual=best_individual,
        best_fitness_result=best_result,
        history=history,
    )
