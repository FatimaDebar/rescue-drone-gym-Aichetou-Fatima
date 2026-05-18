import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from environment.drone_rescue_env    import DroneRescueEnv
from baselines.random_agent          import run_random_agent
from baselines.heuristic_agent       import run_heuristic_agent
from genetic.genetic_algorithm       import genetic_algorithm
from genetic.genetic_agent           import run_genetic_agent

# CONFIGURATION
SEED         = 42
NUM_EPISODES = 30
GRID_SIZE    = 15
NUM_VICTIMS  = 5      
BATTERY_MAX  = 300 
GA_CONFIG = {
    "pop_size"        : 50,
    "generations"     : 40,
    "elite_size"      : 3,
    "tournament_size" : 4,
    "crossover_rate"  : 0.5,
    "mutation_rate"   : 0.15,
    "mutation_std"    : 0.3,
    "n_eval_episodes" : 3,
    "seed"            : 42,
    "verbose"         : True,
}

os.makedirs("results/plots", exist_ok=True)

# ENVIRONNEMENT

env = DroneRescueEnv(
    grid_size   = GRID_SIZE,
    num_victims = NUM_VICTIMS,
    battery_max = BATTERY_MAX
)

# ÉTAPE 1 — ENTRAÎNEMENT DE L'ALGORITHME GÉNÉTIQUE

print("\n" + "=" * 60)
print("ENTRAÎNEMENT — ALGORITHME GÉNÉTIQUE")
print("=" * 60)

best_genome, history = genetic_algorithm(env, **GA_CONFIG)

# ÉTAPE 2 — ÉVALUATION DES 3 AGENTS SUR 30 ÉPISODES

print("\n" + "=" * 60)
print(" AGENT ALÉATOIRE — 30 épisodes")
print("=" * 60)
random_results = run_random_agent(env, num_episodes=NUM_EPISODES, seed=SEED)

print("\n" + "=" * 60)
print(" AGENT HEURISTIQUE — 30 épisodes")
print("=" * 60)
heuristic_results = run_heuristic_agent(env, num_episodes=NUM_EPISODES, seed=SEED)

print("\n" + "=" * 60)
print("AGENT GÉNÉTIQUE — 30 épisodes")
print("=" * 60)
genetic_results = run_genetic_agent(
    env, best_genome, num_episodes=NUM_EPISODES, seed=SEED, verbose=True
)

# ÉTAPE 3 — CALCUL DES STATISTIQUES


def compute_stats(results):
    rewards     = [r["total_reward"]    for r in results]
    victims     = [r["victims_rescued"] for r in results]
    return_rate = sum(1 for r in results if r["returned_to_base"]) / len(results) * 100
    return {
        "reward_mean":  np.mean(rewards),
        "reward_std":   np.std(rewards),
        "reward_min":   np.min(rewards),
        "reward_max":   np.max(rewards),
        "victims_mean": np.mean(victims),
        "return_rate":  return_rate,
        "rewards":      rewards,
        "victims":      victims,
    }

random_stats    = compute_stats(random_results)
heuristic_stats = compute_stats(heuristic_results)
genetic_stats   = compute_stats(genetic_results)

# ÉTAPE 4 — TABLEAU COMPARATIF

print("\n" + "=" * 65)
print(" RÉSULTATS FINAUX — COMPARAISON DES 3 AGENTS")
print("=" * 65)
print(f"{'Métrique':<25} {'Aléatoire':>12} {'Heuristique':>12} {'Génétique':>12}")
print("-" * 65)
metrics = [
    ("Score moyen",      "reward_mean"),
    ("Écart-type",       "reward_std"),
    ("Score min",        "reward_min"),
    ("Score max",        "reward_max"),
    ("Victimes moyennes","victims_mean"),
    ("Taux retour (%)",  "return_rate"),
]
for label, key in metrics:
    print(f"  {label:<23} "
          f"{random_stats[key]:>12.1f} "
          f"{heuristic_stats[key]:>12.1f} "
          f"{genetic_stats[key]:>12.1f}")
print("=" * 65)

# ÉTAPE 5 — GRAPHIQUES

episodes   = list(range(1, NUM_EPISODES + 1))
generations = list(range(1, GA_CONFIG["generations"] + 1))

agents  = ["Aléatoire", "Heuristique", "Génétique"]
colors  = ["red", "blue", "green"]
r_stats = [random_stats, heuristic_stats, genetic_stats]

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle("Drone Secouriste — Comparaison des 3 agents", fontsize=16, fontweight='bold')

# Graphique 1 : Score par épisode 
for stats, label, color in zip(r_stats, agents, colors):
    axes[0, 0].plot(episodes, stats["rewards"], label=label,
                    color=color, marker="o", markersize=3)
    axes[0, 0].axhline(y=stats["reward_mean"], color=color,
                       linestyle="--", alpha=0.4)
axes[0, 0].set_title("Score total par épisode")
axes[0, 0].set_xlabel("Épisode")
axes[0, 0].set_ylabel("Score")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Graphique 2 : Victimes sauvées par épisode 
for stats, label, color in zip(r_stats, agents, colors):
    axes[0, 1].plot(episodes, stats["victims"], label=label,
                    color=color, marker="o", markersize=3)
axes[0, 1].set_title("Victimes sauvées par épisode")
axes[0, 1].set_xlabel("Épisode")
axes[0, 1].set_ylabel("Nombre de victimes")
axes[0, 1].set_ylim(0, NUM_VICTIMS + 1)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Graphique 3 : Taux de retour 
return_rates = [s["return_rate"] for s in r_stats]
bars3 = axes[0, 2].bar(agents, return_rates, color=colors, alpha=0.7)
axes[0, 2].set_title("Taux de retour à la base (%)")
axes[0, 2].set_ylabel("% épisodes avec retour")
axes[0, 2].set_ylim(0, 100)
axes[0, 2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, return_rates):
    axes[0, 2].text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}%", ha='center', fontweight='bold', fontsize=11)

# Graphique 4 : Score moyen ± écart-type
means = [s["reward_mean"] for s in r_stats]
stds  = [s["reward_std"]  for s in r_stats]
bars4 = axes[1, 0].bar(agents, means, yerr=stds, color=colors, alpha=0.7, capsize=8)
axes[1, 0].set_title("Score moyen ± écart-type")
axes[1, 0].set_ylabel("Score moyen")
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars4, means):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, mean + 2,
                    f"{mean:.1f}", ha='center', fontweight='bold')

#Graphique 5 : Victimes moyennes 
victims_means = [s["victims_mean"] for s in r_stats]
bars5 = axes[1, 1].bar(agents, victims_means, color=colors, alpha=0.7)
axes[1, 1].set_title(f"Victimes moyennes / {NUM_VICTIMS}")
axes[1, 1].set_ylabel("Nombre moyen de victimes")
axes[1, 1].set_ylim(0, NUM_VICTIMS)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars5, victims_means):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, mean + 0.05,
                    f"{mean:.1f}", ha='center', fontweight='bold', fontsize=11)

#  Graphique 6 : Distribution des scores (boxplot)
bp = axes[1, 2].boxplot(
    [s["rewards"] for s in r_stats],
    labels=agents,
    patch_artist=True,
    boxprops=dict(facecolor="lightcoral"),
    medianprops=dict(color="black", linewidth=2)
)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
axes[1, 2].set_title("Distribution des scores")
axes[1, 2].set_ylabel("Score")
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Graphique 7 : Courbe de convergence de l'AG
axes[2, 0].plot(generations, history["best_fitness"],
                color="green", label="Meilleur", linewidth=2)
axes[2, 0].plot(generations, history["mean_fitness"],
                color="orange", label="Moyenne", linewidth=1.5, linestyle="--")
axes[2, 0].fill_between(
    generations,
    np.array(history["mean_fitness"]) - np.array(history["std_fitness"]),
    np.array(history["mean_fitness"]) + np.array(history["std_fitness"]),
    alpha=0.2, color="orange", label="±1 std"
)
axes[2, 0].set_title("Convergence de l'algorithme génétique")
axes[2, 0].set_xlabel("Génération")
axes[2, 0].set_ylabel("Fitness")
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

#Graphique 8 : Fitness min/max par génération 
axes[2, 1].plot(generations, history["best_fitness"],
                color="green",  label="Best",  linewidth=2)
axes[2, 1].plot(generations, history["worst_fitness"],
                color="red",    label="Worst", linewidth=1.5, linestyle=":")
axes[2, 1].fill_between(generations,
                         history["worst_fitness"],
                         history["best_fitness"],
                         alpha=0.15, color="green")
axes[2, 1].set_title("Amplitude fitness par génération")
axes[2, 1].set_xlabel("Génération")
axes[2, 1].set_ylabel("Fitness")
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

#Graphique 9 : Radar / synthèse normalisée
categories  = ["Score\nmoyen", "Victimes\nmoyennes", "Taux\nretour"]
all_means   = [[s["reward_mean"], s["victims_mean"], s["return_rate"]]
               for s in r_stats]

# Normalisation 0-1 par métrique
all_means_arr = np.array(all_means)
col_min = all_means_arr.min(axis=0)
col_max = all_means_arr.max(axis=0)
col_rng = col_max - col_min
col_rng[col_rng == 0] = 1
normalized = (all_means_arr - col_min) / col_rng

x_pos = np.arange(len(categories))
width = 0.25

for i, (label, color, norm_vals) in enumerate(zip(agents, colors, normalized)):
    axes[2, 2].bar(x_pos + i * width, norm_vals, width,
                   label=label, color=color, alpha=0.7)

axes[2, 2].set_title("Synthèse normalisée (0 = pire, 1 = meilleur)")
axes[2, 2].set_xticks(x_pos + width)
axes[2, 2].set_xticklabels(categories)
axes[2, 2].set_ylim(0, 1.15)
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3, axis='y')

#Sauvegarde
plt.tight_layout()
output_path = "results/plots/comparaison_3_agents.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nGraphique sauvegardé dans {output_path}")

# ÉTAPE 6 — SAUVEGARDE DU MEILLEUR GÉNOME

genome_path = "results/best_genome.npy"
np.save(genome_path, best_genome)
print(f"Meilleur génome sauvegardé dans {genome_path}")
print(f"   (pour le recharger : genome = np.load('{genome_path}'))")