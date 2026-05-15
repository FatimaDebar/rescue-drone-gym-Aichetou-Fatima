import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from environment.drone_rescue_env import DroneRescueEnv
from baselines.random_agent import run_random_agent
from baselines.heuristic_agent import run_heuristic_agent

np.random.seed(42)

env = DroneRescueEnv(grid_size=15, num_victims=5, battery_max=300)
NUM_EPISODES = 30

# ────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("🎲 AGENT ALÉATOIRE — 30 épisodes")
print("=" * 55)
random_results = run_random_agent(env, num_episodes=NUM_EPISODES, seed=42)

print("\n" + "=" * 55)
print("🧠 AGENT HEURISTIQUE — 30 épisodes")
print("=" * 55)
heuristic_results = run_heuristic_agent(env, num_episodes=NUM_EPISODES, seed=42)

# ────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("📊 RÉSULTATS FINAUX")
print("=" * 55)
print(f"{'Métrique':<25} {'Aléatoire':>12} {'Heuristique':>12}")
print("-" * 55)
print(f"{'Score moyen':<25} {random_stats['reward_mean']:>12.1f} {heuristic_stats['reward_mean']:>12.1f}")
print(f"{'Écart-type':<25} {random_stats['reward_std']:>12.1f} {heuristic_stats['reward_std']:>12.1f}")
print(f"{'Score min':<25} {random_stats['reward_min']:>12.1f} {heuristic_stats['reward_min']:>12.1f}")
print(f"{'Score max':<25} {random_stats['reward_max']:>12.1f} {heuristic_stats['reward_max']:>12.1f}")
print(f"{'Victimes moyennes':<25} {random_stats['victims_mean']:>12.1f} {heuristic_stats['victims_mean']:>12.1f}")
print(f"{'Taux de retour (%)':<25} {random_stats['return_rate']:>12.1f} {heuristic_stats['return_rate']:>12.1f}")
print("=" * 55)

# ────────────────────────────────────────────────────────────────────────────
os.makedirs("results/plots", exist_ok=True)
episodes = list(range(1, NUM_EPISODES + 1))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Drone Secouriste — Comparaison des agents", fontsize=16, fontweight='bold')

# ── Graphique 1 : Score par épisode ─────────────────────────────────────────
axes[0, 0].plot(episodes, random_stats["rewards"],    label="Aléatoire",   color="red",  marker="o", markersize=3)
axes[0, 0].plot(episodes, heuristic_stats["rewards"], label="Heuristique", color="blue", marker="s", markersize=3)
axes[0, 0].axhline(y=random_stats["reward_mean"],    color="red",  linestyle="--", alpha=0.5)
axes[0, 0].axhline(y=heuristic_stats["reward_mean"], color="blue", linestyle="--", alpha=0.5)
axes[0, 0].set_title("Score total par épisode")
axes[0, 0].set_xlabel("Épisode")
axes[0, 0].set_ylabel("Score")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ── Graphique 2 : Victimes sauvées par épisode ──────────────────────────────
axes[0, 1].plot(episodes, random_stats["victims"],    label="Aléatoire",   color="red",  marker="o", markersize=3)
axes[0, 1].plot(episodes, heuristic_stats["victims"], label="Heuristique", color="blue", marker="s", markersize=3)
axes[0, 1].set_title("Victimes sauvées par épisode")
axes[0, 1].set_xlabel("Épisode")
axes[0, 1].set_ylabel("Nombre de victimes")
axes[0, 1].set_ylim(0, 6)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# ── Graphique 3 : Taux de retour ─────────────────────────────────────────────
agents      = ["Aléatoire", "Heuristique"]
return_rates = [random_stats["return_rate"], heuristic_stats["return_rate"]]
colors       = ["red", "blue"]
bars3 = axes[0, 2].bar(agents, return_rates, color=colors, alpha=0.7)
axes[0, 2].set_title("Taux de retour à la base (%)")
axes[0, 2].set_ylabel("% épisodes avec retour")
axes[0, 2].set_ylim(0, 100)
axes[0, 2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars3, return_rates):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, val + 1,
                    f"{val:.0f}%", ha='center', fontweight='bold', fontsize=12)

# ── Graphique 4 : Score moyen ────────────────────────────────────────────────
means = [random_stats["reward_mean"], heuristic_stats["reward_mean"]]
stds  = [random_stats["reward_std"],  heuristic_stats["reward_std"]]
bars4 = axes[1, 0].bar(agents, means, yerr=stds, color=colors, alpha=0.7, capsize=8)
axes[1, 0].set_title("Score moyen ± écart-type")
axes[1, 0].set_ylabel("Score moyen")
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars4, means):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, mean + 2,
                    f"{mean:.1f}", ha='center', fontweight='bold')

# ── Graphique 5 : Victimes moyennes ──────────────────────────────────────────
victims_means = [random_stats["victims_mean"], heuristic_stats["victims_mean"]]
bars5 = axes[1, 1].bar(agents, victims_means, color=colors, alpha=0.7)
axes[1, 1].set_title("Victimes moyennes sauvées / 5")
axes[1, 1].set_ylabel("Nombre moyen de victimes")
axes[1, 1].set_ylim(0, 5)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars5, victims_means):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, mean + 0.05,
                    f"{mean:.1f}", ha='center', fontweight='bold', fontsize=12)

# ── Graphique 6 : Distribution des scores (boxplot) ──────────────────────────
axes[1, 2].boxplot(
    [random_stats["rewards"], heuristic_stats["rewards"]],
    labels=["Aléatoire", "Heuristique"],
    patch_artist=True,
    boxprops=dict(facecolor="lightcoral"),
    medianprops=dict(color="black", linewidth=2)
)
axes[1, 2].set_title("Distribution des scores")
axes[1, 2].set_ylabel("Score")
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/plots/comparaison_agents.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Graphique sauvegardé dans results/plots/comparaison_agents.png")