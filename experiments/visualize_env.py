"""
visualize_drone.py — Simulation Visuelle Jeu Vidéo — Drone Secouriste
======================================================================
Placez ce fichier dans : experiments/visualize_drone.py
Lancez avec           : python experiments/visualize_drone.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')   # essayez 'TkAgg' si Qt5Agg ne marche pas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import time

from environment.drone_rescue_env import DroneRescueEnv

# ── Chargement génome AG ──────────────────────────────────────────────────────
try:
    from genetic.genome import select_action
    GENOME_PATH = os.path.join(os.path.dirname(__file__), 'results', 'best_genome.npy')
    if os.path.exists(GENOME_PATH):
        BEST_GENOME = np.load(GENOME_PATH)
        AGENT_NAME  = "Agent Genetique (AG)"
        print(f"[OK] Genome charge : {GENOME_PATH}")
    else:
        BEST_GENOME = None
        AGENT_NAME  = "Agent Aleatoire"
        print("[!]  Genome introuvable — mode aleatoire")
except Exception as e:
    BEST_GENOME = None
    AGENT_NAME  = "Agent Aleatoire"
    print(f"[!]  Erreur : {e}")

CELL_COLORS = ['#F7F9FC','#2D3748','#FC8181','#48BB78','#F6AD55','#4299E1']
CMAP        = ListedColormap(CELL_COLORS)


def get_action(obs, env):
    if BEST_GENOME is not None:
        try:    return select_action(BEST_GENOME, obs, env=env)
        except: return select_action(BEST_GENOME, obs)
    return env.action_space.sample()


def draw_frame(fig, ax_grid, ax_info, ax_hist,
               env, total_reward, episode, max_episodes,
               score_history, step_rewards):

    # GRILLE
    ax_grid.clear()
    ax_grid.set_facecolor('#1A202C')
    ax_grid.imshow(env.grid.astype(float), cmap=CMAP,
                   vmin=0, vmax=5, interpolation='nearest', aspect='equal')
    n = env.grid_size
    for i in range(n+1):
        ax_grid.axhline(i-0.5, color='#4A5568', linewidth=0.3, alpha=0.5)
        ax_grid.axvline(i-0.5, color='#4A5568', linewidth=0.3, alpha=0.5)
    labels = {5:('D','white'), 2:('V','#742A2A'), 3:('B','#276749'),
              4:('R','#7B341E'), 1:('#','#718096')}
    for r in range(n):
        for c in range(n):
            cell = env.grid[r, c]
            if cell in labels:
                txt, col = labels[cell]
                ax_grid.text(c, r, txt, ha='center', va='center',
                             fontsize=8, color=col, fontweight='bold')
    ax_grid.set_title(
        f'Episode {episode}/{max_episodes}   |   Step {env.steps}/{env.max_steps}',
        color='white', fontsize=12, fontweight='bold', pad=8)
    ax_grid.tick_params(colors='#718096', labelsize=7)
    for sp in ax_grid.spines.values(): sp.set_edgecolor('#4A5568')
    legend_items = [
        mpatches.Patch(color=CELL_COLORS[5], label='D = Drone'),
        mpatches.Patch(color=CELL_COLORS[2], label='V = Victime'),
        mpatches.Patch(color=CELL_COLORS[3], label='B = Base'),
        mpatches.Patch(color=CELL_COLORS[4], label='R = Recharge'),
        mpatches.Patch(color=CELL_COLORS[1], label='# = Obstacle'),
    ]
    ax_grid.legend(handles=legend_items, loc='upper right',
                   fontsize=7, facecolor='#2D3748', labelcolor='white', framealpha=0.9)

    # PANNEAU INFO
    ax_info.clear()
    ax_info.set_facecolor('#2D3748')
    ax_info.axis('off')
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)

    bat_pct   = max(0.0, env.battery / env.battery_max)
    bat_color = '#48BB78' if bat_pct > 0.5 else '#F6AD55' if bat_pct > 0.25 else '#FC8181'
    step_pct  = env.steps / env.max_steps

    ax_info.text(0.5, 0.97, 'DRONE SECOURISTE', ha='center',
                 color='#4299E1', fontsize=13, fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.5, 0.92, AGENT_NAME, ha='center',
                 color='#A0AEC0', fontsize=9, style='italic', transform=ax_info.transAxes)

    # Batterie
    ax_info.text(0.05, 0.87, 'BATTERIE', color='#A0AEC0', fontsize=9,
                 fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.95, 0.87, f'{env.battery}/{env.battery_max} ({bat_pct*100:.0f}%)',
                 color=bat_color, fontsize=9, ha='right', transform=ax_info.transAxes)
    ax_info.add_patch(mpatches.FancyBboxPatch((0.05,0.83), 0.90, 0.030,
        boxstyle='round,pad=0.003', facecolor='#4A5568', transform=ax_info.transAxes, zorder=2))
    if bat_pct > 0:
        ax_info.add_patch(mpatches.FancyBboxPatch((0.05,0.83), 0.90*bat_pct, 0.030,
            boxstyle='round,pad=0.003', facecolor=bat_color, transform=ax_info.transAxes, zorder=3))

    # Victimes
    ax_info.text(0.05, 0.78, 'VICTIMES SAUVEES', color='#A0AEC0', fontsize=9,
                 fontweight='bold', transform=ax_info.transAxes)
    vic_color = '#48BB78' if env.victims_rescued == env.num_victims else 'white'
    ax_info.text(0.5, 0.71, f'{env.victims_rescued}  /  {env.num_victims}',
                 color=vic_color, fontsize=28, fontweight='bold',
                 ha='center', transform=ax_info.transAxes)
    for i in range(env.num_victims):
        x_pos = 0.15 + i * (0.70 / max(env.num_victims-1, 1))
        color = '#48BB78' if i < env.victims_rescued else '#FC8181'
        ax_info.add_patch(mpatches.Circle((x_pos, 0.635), 0.045, color=color,
            transform=ax_info.transAxes, zorder=3))
        ax_info.text(x_pos, 0.635, 'S' if i < env.victims_rescued else 'X',
            ha='center', va='center', fontsize=8, color='white', fontweight='bold',
            transform=ax_info.transAxes)

    # Score
    ax_info.text(0.05, 0.58, 'SCORE CUMULE', color='#A0AEC0', fontsize=9,
                 fontweight='bold', transform=ax_info.transAxes)
    sc_color = '#F6AD55' if total_reward >= 0 else '#FC8181'
    ax_info.text(0.5, 0.51, f'{total_reward:+.0f}', color=sc_color, fontsize=26,
                 fontweight='bold', ha='center', transform=ax_info.transAxes)

    # Steps
    ax_info.text(0.05, 0.45, 'PROGRESSION', color='#A0AEC0', fontsize=9,
                 fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.95, 0.45, f'{env.steps}/{env.max_steps}', color='white',
                 fontsize=9, ha='right', transform=ax_info.transAxes)
    ax_info.add_patch(mpatches.FancyBboxPatch((0.05,0.41), 0.90, 0.028,
        boxstyle='round,pad=0.003', facecolor='#4A5568', transform=ax_info.transAxes, zorder=2))
    if step_pct > 0:
        ax_info.add_patch(mpatches.FancyBboxPatch((0.05,0.41), 0.90*step_pct, 0.028,
            boxstyle='round,pad=0.003', facecolor='#667EEA', transform=ax_info.transAxes, zorder=3))

    # Historique
    ax_info.text(0.5, 0.365, 'SCORES PRECEDENTS', ha='center',
                 color='#718096', fontsize=8, transform=ax_info.transAxes)
    if score_history:
        last = score_history[-5:]
        for i, sc in enumerate(last):
            col = '#48BB78' if sc >= 0 else '#FC8181'
            ep_num = episode - len(last) + i
            ax_info.text(0.5, 0.325 - i*0.052,
                         f'Ep {ep_num} : {sc:+.0f}', ha='center',
                         color=col, fontsize=8, transform=ax_info.transAxes)

    # HISTOGRAMME
    ax_hist.clear()
    ax_hist.set_facecolor('#1A202C')
    if len(step_rewards) > 1:
        cols = ['#48BB78' if r >= 0 else '#FC8181' for r in step_rewards]
        ax_hist.bar(range(len(step_rewards)), step_rewards, color=cols, width=1.0, alpha=0.8)
        ax_hist.axhline(0, color='white', linewidth=0.5, alpha=0.5)
    ax_hist.set_title('Reward par step (episode courant)',
                      color='white', fontsize=9, pad=4)
    ax_hist.tick_params(colors='#718096', labelsize=6)
    for sp in ax_hist.spines.values(): sp.set_edgecolor('#4A5568')


def run_simulation(seed=42, speed=0.05, max_episodes=5):
    env = DroneRescueEnv(grid_size=15, num_victims=5, battery_max=300)

    fig = plt.figure(figsize=(16, 9), facecolor='#1A202C')
    fig.suptitle(f'SIMULATION — Drone Secouriste   [{AGENT_NAME}]',
                 fontsize=14, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[2.2, 1], height_ratios=[3, 1],
                           left=0.04, right=0.97, top=0.94, bottom=0.04,
                           wspace=0.10, hspace=0.20)
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, :])

    plt.ion()
    plt.show()

    score_history = []
    all_stats     = []

    for episode in range(1, max_episodes + 1):
        obs, _       = env.reset(seed=seed + episode * 13)
        total_reward = 0.0
        terminated   = False
        truncated    = False
        step_rewards = []
        print(f"\n--- Episode {episode}/{max_episodes} ---")

        while not terminated and not truncated:
            draw_frame(fig, ax_grid, ax_info, ax_hist,
                       env, total_reward, episode, max_episodes,
                       score_history, step_rewards)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(speed)

            action = get_action(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_rewards.append(reward)

        # Dernière frame
        draw_frame(fig, ax_grid, ax_info, ax_hist,
                   env, total_reward, episode, max_episodes,
                   score_history, step_rewards)
        fig.canvas.draw()
        fig.canvas.flush_events()

        score_history.append(total_reward)
        returned = terminated and info['battery'] > 0
        status   = ('MISSION COMPLETE !' if info['victims_rescued'] == env.num_victims and returned
                    else 'Batterie vide'   if info['battery'] <= 0
                    else 'Retour base'     if returned
                    else 'Temps ecoule')

        all_stats.append({'episode': episode, 'reward': total_reward,
                          'victims': info['victims_rescued'], 'returned': returned,
                          'steps': info['steps']})

        print(f"  {status:20s} | Victimes: {info['victims_rescued']}/{env.num_victims} | "
              f"Score: {total_reward:+7.1f} | Steps: {info['steps']}")
        time.sleep(1.0)

    rewards  = [s['reward']   for s in all_stats]
    victims  = [s['victims']  for s in all_stats]
    returned = [s['returned'] for s in all_stats]
    print(f"\n{'='*50}")
    print(f"  RESUME : Score moy={np.mean(rewards):+.1f} | "
          f"Victimes={np.mean(victims):.1f}/{env.num_victims} | "
          f"Retour={sum(returned)/len(returned)*100:.0f}%")
    print(f"{'='*50}")

    plt.ioff()
    plt.show()
    env.close()


if __name__ == '__main__':
    print("=" * 55)
    print("  DRONE SECOURISTE — Simulation Visuelle")
    print("=" * 55)
    print("  speed=0.03 rapide | speed=0.15 lent")
    print("=" * 55)

    run_simulation(
        seed         = 42,
        speed        = 0.01,   # secondes entre chaque frame
        max_episodes = 3  )     # episodes a visualiser