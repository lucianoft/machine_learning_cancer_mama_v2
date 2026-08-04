import random

from ga.individual import random_individual
from ga.operators import crossover, mutate, tournament_selection
from ga.search_space import SEARCH_SPACES

MODEL_FAMILY = "random_forest"


def test_random_individual_respeita_limites():
    rng = random.Random(1)
    for _ in range(50):
        individual = random_individual(MODEL_FAMILY, rng)
        _assert_within_bounds(individual)


def test_crossover_filhos_so_tem_genes_dos_pais():
    rng = random.Random(2)
    parent_a = random_individual(MODEL_FAMILY, rng)
    parent_b = random_individual(MODEL_FAMILY, rng)

    child_a, child_b = crossover(parent_a, parent_b, MODEL_FAMILY, rng, crossover_rate=1.0)

    for gene in SEARCH_SPACES[MODEL_FAMILY]:
        assert child_a[gene] in (parent_a[gene], parent_b[gene])
        assert child_b[gene] in (parent_a[gene], parent_b[gene])


def test_crossover_rate_zero_clona_pais():
    rng = random.Random(3)
    parent_a = random_individual(MODEL_FAMILY, rng)
    parent_b = random_individual(MODEL_FAMILY, rng)

    child_a, child_b = crossover(parent_a, parent_b, MODEL_FAMILY, rng, crossover_rate=0.0)

    assert child_a == parent_a
    assert child_b == parent_b


def test_mutate_mantem_valores_dentro_dos_limites():
    rng = random.Random(4)
    individual = random_individual(MODEL_FAMILY, rng)

    for _ in range(50):
        individual = mutate(individual, MODEL_FAMILY, rng, mutation_rate=0.8)
        _assert_within_bounds(individual)


def test_mutation_rate_zero_nao_muda_nada():
    rng = random.Random(5)
    individual = random_individual(MODEL_FAMILY, rng)
    mutated = mutate(individual, MODEL_FAMILY, rng, mutation_rate=0.0)
    assert mutated == individual


def test_tournament_selection_favorece_maior_fitness():
    rng = random.Random(6)
    population = [{"id": i} for i in range(10)]
    fitnesses = [float(i) for i in range(10)]  # indice 9 é sempre o melhor

    vitorias_do_melhor = 0
    for _ in range(200):
        selecionado = tournament_selection(population, fitnesses, rng, k=3)
        if selecionado["id"] == 9:
            vitorias_do_melhor += 1

    # com k=3 numa populacao de 10, o melhor individuo deve vencer bem mais que 1/10
    # das vezes (frequencia esperada por acaso)
    assert vitorias_do_melhor > 20


def _assert_within_bounds(individual: dict) -> None:
    space = SEARCH_SPACES[MODEL_FAMILY]
    for gene, spec in space.items():
        kind = spec[0]
        value = individual[gene]
        if kind == "int":
            _, low, high = spec
            assert low <= value <= high
        elif kind == "log_float":
            _, low, high = spec
            assert low <= value <= high
        elif kind == "categorical":
            _, choices = spec
            assert value in choices
