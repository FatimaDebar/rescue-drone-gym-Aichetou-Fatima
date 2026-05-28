import pygame
import numpy as np
import sys
import os

# Chemins corrects selon votre structure
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'genetic'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'baselines'))

from drone_rescue_env import DroneRescueEnv
from genetic_agent import GeneticAgent

# ─── Couleurs ───────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0  )
GRAY       = (200, 200, 200)
DARK_GRAY  = (30,  30,  30 )
DARK_BG    = (20,  20,  40 )
RED        = (220, 50,  50 )
GREEN      = (50,  200, 50 )
BLUE       = (50,  120, 220)
YELLOW     = (255, 220, 0  )
ORANGE     = (255, 140, 0  )
PURPLE     = (150, 50,  200)
LIGHT_BLUE = (135, 206, 235)
DARK_GREEN = (0,   120, 0  )
TEAL       = (0,   180, 180)

# ─── Dimensions ─────────────────────────────────────────
CELL_SIZE  = 44
MARGIN     = 10
INFO_W     = 260
FPS        = 8


def load_best_genome():
    genome_path = os.path.join(
        os.path.dirname(__file__), 'experiments', 'best_genome.npy'
    )
    if os.path.exists(genome_path):
        print(f"Genome charge : {genome_path}")
        return np.load(genome_path)
    else:
        print("Aucun genome trouve.")
        print("Lancez d'abord : python experiments/run_genetic.py")
        return None


def draw_grid(screen, env, font_cell):
    """Dessine la grille de jeu."""
    grid_size = env.grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            x = MARGIN + col * CELL_SIZE
            y = MARGIN + row * CELL_SIZE
            val = env.grid[row, col]

            if val == 1:
                color = (15, 15, 15)
            elif val == 2:
                color = (180, 30, 30)
            elif val == 3:
                color = (30, 160, 30)
            elif val == 4:
                color = (200, 110, 0)
            elif val == 5:
                color = (40, 100, 200)
            else:
                color = (240, 240, 245)

            pygame.draw.rect(
                screen, color,
                (x+1, y+1, CELL_SIZE-2, CELL_SIZE-2),
                border_radius=5
            )

            symbols    = {1: "#", 2: "V", 3: "B", 4: "R", 5: "D"}
            txt_colors = {1: GRAY, 2: WHITE, 3: WHITE, 4: WHITE, 5: WHITE}
            if val in symbols:
                s  = font_cell.render(symbols[val], True, txt_colors[val])
                sx = x + CELL_SIZE//2 - s.get_width()//2
                sy = y + CELL_SIZE//2 - s.get_height()//2
                screen.blit(s, (sx, sy))


def draw_info_panel(screen, env, episode, total_episodes,
                    font, font_sm, font_title, info_x, height):
    """Dessine le panneau d'informations à droite."""

    y = MARGIN + 10

    title = font_title.render("DRONE SECOURISTE", True, YELLOW)
    screen.blit(title, (info_x, y))
    y += 35

    subtitle = font_sm.render("Simulation Agent Genetique", True, TEAL)
    screen.blit(subtitle, (info_x, y))
    y += 30

    pygame.draw.line(screen, GRAY,
                     (info_x, y), (info_x + INFO_W - 20, y), 1)
    y += 15

    ep_text = font.render(
        f"Episode : {episode} / {total_episodes}", True, WHITE)
    screen.blit(ep_text, (info_x, y))
    y += 30

    step_text = font.render(
        f"Steps   : {env.steps} / {env.max_steps}", True, WHITE)
    screen.blit(step_text, (info_x, y))
    y += 30

    pygame.draw.line(screen, GRAY,
                     (info_x, y), (info_x + INFO_W - 20, y), 1)
    y += 15

    bat_pct = max(0, env.battery / env.battery_max)
    if bat_pct > 0.5:
        bat_color = GREEN
    elif bat_pct > 0.2:
        bat_color = ORANGE
    else:
        bat_color = RED

    bat_lbl = font.render(
        f"Batterie : {env.battery}/{env.battery_max}",
        True, bat_color)
    screen.blit(bat_lbl, (info_x, y))
    y += 22

    bar_w = INFO_W - 30
    pygame.draw.rect(screen, GRAY,
                     (info_x, y, bar_w, 14), border_radius=5)
    pygame.draw.rect(screen, bat_color,
                     (info_x, y, int(bar_w * bat_pct), 14),
                     border_radius=5)
    y += 28

    vic_color = GREEN if env.victims_rescued == env.num_victims else WHITE
    vic_text  = font.render(
        f"Victimes : {env.victims_rescued} / {env.num_victims}",
        True, vic_color)
    screen.blit(vic_text, (info_x, y))
    y += 25

    for i in range(env.num_victims):
        color = GREEN if i < env.victims_rescued else RED
        pygame.draw.circle(
            screen, color,
            (info_x + 12 + i * 28, y + 10), 10)
        lbl = font_sm.render(
            "v" if i < env.victims_rescued else "V",
            True, WHITE)
        screen.blit(lbl, (info_x + 6 + i * 28, y + 2))
    y += 35

    pygame.draw.line(screen, GRAY,
                     (info_x, y), (info_x + INFO_W - 20, y), 1)
    y += 15

    score_color = GREEN if env.total_reward >= 0 else RED
    score_text  = font.render(
        f"Score : {env.total_reward:.0f}", True, score_color)
    screen.blit(score_text, (info_x, y))
    y += 35

    pygame.draw.line(screen, GRAY,
                     (info_x, y), (info_x + INFO_W - 20, y), 1)
    y += 15

    leg_title = font.render("Legende :", True, YELLOW)
    screen.blit(leg_title, (info_x, y))
    y += 28

    legende = [
        ((40,  100, 200), "D - Drone"),
        ((180, 30,  30 ), "V - Victime"),
        ((30,  160, 30 ), "B - Base"),
        ((200, 110, 0  ), "R - Recharge"),
        ((15,  15,  15 ), "# - Obstacle"),
    ]
    for color, text in legende:
        pygame.draw.rect(screen, color,
                         (info_x, y, 18, 18), border_radius=3)
        pygame.draw.rect(screen, GRAY,
                         (info_x, y, 18, 18), 1, border_radius=3)
        t = font_sm.render(text, True, WHITE)
        screen.blit(t, (info_x + 26, y + 1))
        y += 26

    y += 10
    pygame.draw.line(screen, GRAY,
                     (info_x, y), (info_x + INFO_W - 20, y), 1)
    y += 15

    ctrl_title = font.render("Controles :", True, YELLOW)
    screen.blit(ctrl_title, (info_x, y))
    y += 28

    controles = [
        ("ESPACE", "Pause / Reprendre"),
        ("->",     "Step manuel"),
        ("R",      "Nouvel episode"),
        ("+ / -",  "Vitesse"),
        ("ESC",    "Quitter"),
    ]
    for key, desc in controles:
        k = font_sm.render(f"{key:<8}", True, LIGHT_BLUE)
        d = font_sm.render(desc, True, GRAY)
        screen.blit(k, (info_x, y))
        screen.blit(d, (info_x + 75, y))
        y += 22


def draw_message(screen, message, color, font_big, grid_w, grid_h):
    """Affiche un message centre sur la grille."""
    surf = font_big.render(message, True, color)
    bg   = pygame.Surface(
        (surf.get_width() + 30, surf.get_height() + 20))
    bg.set_alpha(210)
    bg.fill(DARK_BG)
    mx = grid_w // 2 - surf.get_width()  // 2
    my = grid_h // 2 - surf.get_height() // 2
    screen.blit(bg,   (mx - 15, my - 10))
    screen.blit(surf, (mx, my))


def run_simulation():
    pygame.init()

    # ── Chargement du genome ──
    genome = load_best_genome()
    if genome is None:
        pygame.quit()
        return

    # ── Initialisation environnement ──
    env = DroneRescueEnv(grid_size=15, num_victims=5, battery_max=200)

    # ── Initialisation agent — CORRECTION ICI ──
    agent = GeneticAgent(genome)   # <-- 1 seul argument : le genome

    # ── Fenetre ──
    grid_size = env.grid_size
    grid_w    = MARGIN + grid_size * CELL_SIZE + MARGIN
    grid_h    = MARGIN + grid_size * CELL_SIZE + MARGIN
    win_w     = grid_w + INFO_W + MARGIN
    win_h     = grid_h

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        "Drone Secouriste — Simulation Agent Genetique")

    font_title = pygame.font.SysFont("Arial", 17, bold=True)
    font       = pygame.font.SysFont("Arial", 15, bold=True)
    font_sm    = pygame.font.SysFont("Arial", 14)
    font_cell  = pygame.font.SysFont("Arial", 16, bold=True)
    font_big   = pygame.font.SysFont("Arial", 30, bold=True)

    clock  = pygame.time.Clock()
    fps    = FPS
    paused = False

    total_episodes = 10
    episode        = 1
    obs, _         = env.reset(seed=episode)
    game_over      = False
    end_message    = ""
    end_color      = WHITE
    info_x         = grid_w + MARGIN

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_SPACE:
                    paused = not paused

                # Step manuel — CORRECTION ICI
                if (event.key == pygame.K_RIGHT
                        and paused and not game_over):
                    action = agent.select_action(obs, env=env)  # <-- CORRIGE
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        game_over = True
                        if (env.victims_rescued == env.num_victims
                                and terminated):
                            end_message = "MISSION ACCOMPLIE !"
                            end_color   = GREEN
                        elif env.battery <= 0:
                            end_message = "BATTERIE EPUISEE !"
                            end_color   = RED
                        else:
                            end_message = "TEMPS ECOULE !"
                            end_color   = ORANGE

                if event.key == pygame.K_r:
                    obs, _      = env.reset(seed=episode)
                    game_over   = False
                    end_message = ""
                    paused      = False

                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    fps = min(30, fps + 2)

                if event.key == pygame.K_MINUS:
                    fps = max(1, fps - 2)

        # ── Action automatique — CORRECTION ICI ──
        if not paused and not game_over:
            action = agent.select_action(obs, env=env)  # <-- CORRIGE
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                game_over = True
                if (env.victims_rescued == env.num_victims
                        and terminated):
                    end_message = "MISSION ACCOMPLIE !"
                    end_color   = GREEN
                elif env.battery <= 0:
                    end_message = "BATTERIE EPUISEE !"
                    end_color   = RED
                else:
                    end_message = "TEMPS ECOULE !"
                    end_color   = ORANGE

        # ── Passage automatique au prochain episode ──
        if game_over:
            screen.fill(DARK_BG)
            draw_grid(screen, env, font_cell)
            draw_info_panel(
                screen, env, episode, total_episodes,
                font, font_sm, font_title, info_x, win_h
            )
            draw_message(
                screen, end_message, end_color,
                font_big, grid_w, grid_h
            )
            pygame.display.flip()
            pygame.time.wait(2500)

            episode += 1
            if episode > total_episodes:
                episode = 1
            obs, _      = env.reset(seed=episode)
            game_over   = False
            end_message = ""

        # ── Dessin principal ──
        screen.fill(DARK_BG)
        draw_grid(screen, env, font_cell)

        pygame.draw.line(
            screen, GRAY,
            (grid_w, MARGIN),
            (grid_w, win_h - MARGIN), 1
        )

        draw_info_panel(
            screen, env, episode, total_episodes,
            font, font_sm, font_title, info_x, win_h
        )

        fps_txt = font_sm.render(
            f"Vitesse : {fps} FPS", True, LIGHT_BLUE)
        screen.blit(fps_txt, (info_x, win_h - 30))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    run_simulation()