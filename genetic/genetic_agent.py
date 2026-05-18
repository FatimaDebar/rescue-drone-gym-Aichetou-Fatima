import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic.genome import select_action, evaluate_genome, GENOME_SIZE


class GeneticAgent:

    def __init__(self, genome):
        self.genome = genome
        self.total_reward   = 0.0
        self.victims_rescued = 0
        self.returned_to_base = False
        self.steps = 0

    def select_action(self, observation):

        return select_action(self.genome, observation)

    def reset_stats(self):
        self.total_reward    = 0.0
        self.victims_rescued = 0
        self.returned_to_base = False
        self.steps = 0

# Exécution sur plusieurs épisodes (même interface que random/heuristic)


def run_genetic_agent(env, genome, num_episodes=30, seed=42, verbose=True):
    np.random.seed(seed)
    agent = GeneticAgent(genome)
    results = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        agent.reset_stats()
        terminated = False
        truncated  = False

        while not terminated and not truncated:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.total_reward    += reward
            agent.victims_rescued  = info["victims_rescued"]
            agent.steps            = info["steps"]

        # Le drone est rentré en base si terminé avec batterie restante
        agent.returned_to_base = terminated and info["battery"] > 0

        results.append({
            "episode":          episode + 1,
            "total_reward":     agent.total_reward,
            "victims_rescued":  agent.victims_rescued,
            "returned_to_base": agent.returned_to_base,
            "steps":            agent.steps
        })

        if verbose:
            print(f"[Génétique] Épisode {episode+1:2d} | "
                  f"Reward: {agent.total_reward:6.1f} | "
                  f"Victimes: {agent.victims_rescued}/{env.num_victims} | "
                  f"Retour: {'Ok' if agent.returned_to_base else 'Non'} | "
                  f"Steps: {agent.steps}")

    return results

# Test rapide (si exécuté directement)

if __name__ == "__main__":
    from environment.drone_rescue_env import DroneRescueEnv
    from genetic.genome import random_genome, genome_info

    genome_info()

    env = DroneRescueEnv(grid_size=15, num_victims=5, battery_max=300)

    # Test avec un génome aléatoire (avant entraînement)
    g = random_genome(np.random.default_rng(42))
    print("\n Test de l'agent génétique (génome aléatoire — avant AG)\n")
    results = run_genetic_agent(env, g, num_episodes=5, seed=42)

    rewards  = [r["total_reward"]    for r in results]
    victims  = [r["victims_rescued"] for r in results]
    returned = [r["returned_to_base"] for r in results]

    print("\n" + "=" * 50)
    print(" RÉSUMÉ (génome aléatoire)")
    print("=" * 50)
    print(f"  Score moyen    : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Victimes moy.  : {np.mean(victims):.1f} / {env.num_victims}")
    print(f"  Taux de retour : {sum(returned)/len(returned)*100:.0f}%")
    print("=" * 50)
    print("\n GeneticAgent opérationnel — prêt pour l'algorithme génétique.")