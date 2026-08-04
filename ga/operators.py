"""Operadores geneticos: selecao por torneio, crossover uniforme por gene, mutacao por
resample dentro dos limites do gene. Cada funcao e pura (recebe RNG explicito) pra dar
pra testar de forma determinística com seed fixo.
"""

import random

from ga.individual import Individual, clip_gene, random_gene
from ga.search_space import SEARCH_SPACES


def tournament_selection(
    population: list[Individual],
    fitnesses: list[float],
    rng: random.Random,
    k: int = 3,
) -> Individual:
    """Sorteia k individuos e devolve o de maior fitness (maximizacao)."""
    contenders_idx = rng.sample(range(len(population)), k)
    best_idx = max(contenders_idx, key=lambda i: fitnesses[i])
    return dict(population[best_idx])


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    model_family: str,
    rng: random.Random,
    crossover_rate: float,
) -> tuple[Individual, Individual]:
    """Crossover uniforme: cada gene do filho vem de um dos pais com 50% de chance.
    Com probabilidade (1 - crossover_rate), os pais passam direto (clonados)."""
    if rng.random() > crossover_rate:
        return dict(parent_a), dict(parent_b)

    genes = SEARCH_SPACES[model_family].keys()
    child_a, child_b = {}, {}
    for gene in genes:
        if rng.random() < 0.5:
            child_a[gene], child_b[gene] = parent_a[gene], parent_b[gene]
        else:
            child_a[gene], child_b[gene] = parent_b[gene], parent_a[gene]
    return child_a, child_b


def mutate(
    individual: Individual,
    model_family: str,
    rng: random.Random,
    mutation_rate: float,
) -> Individual:
    """Cada gene tem `mutation_rate` de chance de ser resortido dentro dos limites
    do espaco de busca (mutacao por resample, nao perturbacao incremental — mantem
    simples e sempre gera valores validos)."""
    space = SEARCH_SPACES[model_family]
    mutated = dict(individual)
    for gene, spec in space.items():
        if rng.random() < mutation_rate:
            mutated[gene] = clip_gene(spec, random_gene(spec, rng))
    return mutated
