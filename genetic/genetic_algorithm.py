import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic.genome import (
    init_population, evaluate_genome,
    clone_genome, GENOME_SIZE
)

# 1. SÉLECTION PAR TOURNOI


def tournament_selection(population, fitness_scores, tournament_size=4, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
    return clone_genome(population[best_idx])

# 2. CROISEMENT UNIFORME


def uniform_crossover(parent1, parent2, crossover_rate=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mask   = rng.random(GENOME_SIZE) < crossover_rate
    child1 = np.where(mask, parent1, parent2).astype(np.float32)
    child2 = np.where(mask, parent2, parent1).astype(np.float32)
    return child1, child2

# 3. MUTATION GAUSSIENNE ADAPTATIVE

def mutate(genome, mutation_rate=0.1, mutation_std=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    child = clone_genome(genome)
    mask  = rng.random(GENOME_SIZE) < mutation_rate
    noise = rng.normal(0, mutation_std, GENOME_SIZE).astype(np.float32)
    child[mask] += noise[mask]
    return child

# 4. ÉLITISME

def elitism(population, fitness_scores, elite_size=2):
    sorted_idx = np.argsort(fitness_scores)[::-1]
    return [clone_genome(population[i]) for i in sorted_idx[:elite_size]]

# 5. BOUCLE PRINCIPALE DE L'ALGORITHME GÉNÉTIQUE

def genetic_algorithm(
    env,
    pop_size        = 50,
    generations     = 40,
    elite_size      = 3,
    tournament_size = 4,
    crossover_rate  = 0.5,
    mutation_rate   = 0.15,
    mutation_std    = 0.3,
    n_eval_episodes = 3,
    seed            = 42,
    verbose         = True
):
    
    rng = np.random.default_rng(seed)

    # Initialisation
    population = init_population(pop_size, seed=seed)

    history = {
        "best_fitness":  [],
        "mean_fitness":  [],
        "std_fitness":   [],
        "worst_fitness": []
    }

    best_genome       = None
    best_fitness_ever = -np.inf

    if verbose:
        print("=" * 60)
        print("ALGORITHME GÉNÉTIQUE — Drone Secouriste")
        print("=" * 60)
        print(f"  Population     : {pop_size} individus")
        print(f"  Générations    : {generations}")
        print(f"  Élitisme       : {elite_size} meilleurs conservés")
        print(f"  Tournoi        : taille {tournament_size}")
        print(f"  Mutation rate  : {mutation_rate}")
        print(f"  Mutation std   : {mutation_std} (décroissant)")
        print(f"  Éval épisodes  : {n_eval_episodes} par individu")
        print("=" * 60)

    # Boucle évolutionnaire 
    for gen in range(generations):

        # Mutation std décroissante : exploration -> exploitation
        current_std = mutation_std * (1.0 - 0.75 * gen / generations)

        # Évaluation fitness
        fitness_scores = [
            evaluate_genome(genome, env, n_episodes=n_eval_episodes, seed=seed + gen)
            for genome in population
        ]

        fitness_scores = np.array(fitness_scores)

        # Statistiques de la génération 
        best_idx      = int(np.argmax(fitness_scores))
        gen_best      = fitness_scores[best_idx]
        gen_mean      = float(np.mean(fitness_scores))
        gen_std       = float(np.std(fitness_scores))
        gen_worst     = float(np.min(fitness_scores))

        history["best_fitness"].append(gen_best)
        history["mean_fitness"].append(gen_mean)
        history["std_fitness"].append(gen_std)
        history["worst_fitness"].append(gen_worst)

        # Mise à jour du meilleur global 
        if gen_best > best_fitness_ever:
            best_fitness_ever = gen_best
            best_genome       = clone_genome(population[best_idx])

        if verbose:
            print(f"  Gén {gen+1:3d}/{generations} | "
                  f"Best: {gen_best:7.1f} | "
                  f"Moy: {gen_mean:7.1f} | "
                  f"Std: {gen_std:6.1f} | "
                  f"Mut_std: {current_std:.3f}")

        # Élitisme : conserver les meilleurs 
        elites = elitism(population, list(fitness_scores), elite_size)

        # Nouvelle génération par sélection + croisement + mutation
        new_population = elites.copy()

        while len(new_population) < pop_size:
            # Sélection de deux parents par tournoi
            parent1 = tournament_selection(
                population, list(fitness_scores), tournament_size, rng
            )
            parent2 = tournament_selection(
                population, list(fitness_scores), tournament_size, rng
            )

            # Croisement
            child1, child2 = uniform_crossover(
                parent1, parent2, crossover_rate, rng
            )

            # Mutation
            child1 = mutate(child1, mutation_rate, current_std, rng)
            child2 = mutate(child2, mutation_rate, current_std, rng)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    #  Résumé final
    if verbose:
        print("=" * 60)
        print(f" Entraînement terminé !")
        print(f"   Meilleure fitness atteinte : {best_fitness_ever:.1f}")
        print("=" * 60)

    return best_genome, history

# Test rapide (si exécuté directement)

if __name__ == "__main__":
    from environment.drone_rescue_env import DroneRescueEnv

    env = DroneRescueEnv(grid_size=15, num_victims=5, battery_max=300)

    best_genome, history = genetic_algorithm(
        env,
        pop_size        = 20,    
        generations     = 10,
        elite_size      = 2,
        tournament_size = 4,
        mutation_rate   = 0.15,
        mutation_std    = 0.3,
        n_eval_episodes = 2,
        seed            = 42,
        verbose         = True
    )

    print(f"\nTaille du meilleur génome : {best_genome.shape}")
    print(f"Fitness finale            : {history['best_fitness'][-1]:.1f}")
    print(f"Progression best fitness  : "
          f"{history['best_fitness'][0]:.1f} → {history['best_fitness'][-1]:.1f}")