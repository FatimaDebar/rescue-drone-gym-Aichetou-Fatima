import pygame
import numpy as np
from environment.drone_rescue_env import DroneRescueEnv

# Couleurs
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0  )
GRAY       = (200, 200, 200)
DARK_GRAY  = (50,  50,  50 )
RED        = (220, 50,  50 )
GREEN      = (50,  200, 50 )
BLUE       = (50,  100, 220)
YELLOW     = (255, 220, 0  )
ORANGE     = (255, 140, 0  )
PURPLE     = (150, 50,  200)
LIGHT_BLUE = (135, 206, 235)

# Taille des cases
CELL_SIZE  = 45
MARGIN     = 10
INFO_WIDTH = 250

def draw_env(screen, env, font, font_small):
    grid_size = env.grid_size
    width  = grid_size * CELL_SIZE + 2 * MARGIN + INFO_WIDTH
    height = grid_size * CELL_SIZE + 2 * MARGIN

    screen.fill(DARK_GRAY)

    # Dessiner la grille
    for row in range(grid_size):
        for col in range(grid_size):
            x = MARGIN + col * CELL_SIZE
            y = MARGIN + row * CELL_SIZE
            cell_value = env.grid[row, col]

            # Fond de la case
            if cell_value == 1:
                color = BLACK          # Obstacle
            elif cell_value == 2:
                color = RED            # Victime
            elif cell_value == 3:
                color = GREEN          # Base
            elif cell_value == 4:
                color = ORANGE         # Recharge
            elif cell_value == 5:
                color = BLUE           # Drone
            else:
                color = WHITE          # Case vide

            pygame.draw.rect(screen, color,
                             (x, y, CELL_SIZE - 2, CELL_SIZE - 2),
                             border_radius=4)

            # Symboles sur les cases
            if cell_value == 1:
                label = font_small.render("#", True, GRAY)
            elif cell_value == 2:
                label = font_small.render("V", True, WHITE)
            elif cell_value == 3:
                label = font_small.render("B", True, WHITE)
            elif cell_value == 4:
                label = font_small.render("R", True, WHITE)
            elif cell_value == 5:
                label = font_small.render("D", True, WHITE)
            else:
                label = None

            if label:
                lx = x + (CELL_SIZE - 2) // 2 - label.get_width()  // 2
                ly = y + (CELL_SIZE - 2) // 2 - label.get_height() // 2
                screen.blit(label, (lx, ly))

    # Ligne de séparation panneau info
    sep_x = MARGIN + grid_size * CELL_SIZE + MARGIN
    pygame.draw.line(screen, GRAY,
                     (sep_x, MARGIN),
                     (sep_x, height - MARGIN), 2)

    # Panneau d'informations à droite
    info_x = sep_x + 10
    info_y = MARGIN + 10

    # Titre
    title = font.render("DRONE SECOURISTE", True, YELLOW)
    screen.blit(title, (info_x, info_y))
    info_y += 40

    # Ligne séparatrice
    pygame.draw.line(screen, GRAY,
                     (info_x, info_y),
                     (info_x + INFO_WIDTH - 20, info_y), 1)
    info_y += 15

    # Batterie
    bat_pct = env.battery / env.battery_max
    bat_color = GREEN if bat_pct > 0.5 else ORANGE if bat_pct > 0.2 else RED
    bat_text = font.render(f"Batterie : {env.battery}/{env.battery_max}",
                           True, bat_color)
    screen.blit(bat_text, (info_x, info_y))
    info_y += 25

    # Barre de batterie
    bar_w = INFO_WIDTH - 30
    pygame.draw.rect(screen, GRAY,
                     (info_x, info_y, bar_w, 14), border_radius=4)
    pygame.draw.rect(screen, bat_color,
                     (info_x, info_y, int(bar_w * bat_pct), 14),
                     border_radius=4)
    info_y += 30

    # Victimes
    vic_text = font.render(
        f"Victimes : {env.victims_rescued}/{env.num_victims}",
        True, WHITE)
    screen.blit(vic_text, (info_x, info_y))
    info_y += 30

    # Steps
    step_text = font.render(f"Steps : {env.steps}/{env.max_steps}",
                            True, WHITE)
    screen.blit(step_text, (info_x, info_y))
    info_y += 40

    # Ligne séparatrice
    pygame.draw.line(screen, GRAY,
                     (info_x, info_y),
                     (info_x + INFO_WIDTH - 20, info_y), 1)
    info_y += 15

    # Légende
    legende_title = font.render("Légende :", True, YELLOW)
    screen.blit(legende_title, (info_x, info_y))
    info_y += 30

    legende = [
        (BLUE,   "D — Drone"),
        (RED,    "V — Victime"),
        (GREEN,  "B — Base"),
        (ORANGE, "R — Recharge"),
        (BLACK,  "# — Obstacle"),
        (WHITE,  ". — Vide"),
    ]

    for color, text in legende:
        pygame.draw.rect(screen, color,
                         (info_x, info_y, 18, 18), border_radius=3)
        if color == WHITE:
            pygame.draw.rect(screen, GRAY,
                             (info_x, info_y, 18, 18), 1, border_radius=3)
        leg_text = font_small.render(text, True, WHITE)
        screen.blit(leg_text, (info_x + 26, info_y + 1))
        info_y += 28

    info_y += 10
    pygame.draw.line(screen, GRAY,
                     (info_x, info_y),
                     (info_x + INFO_WIDTH - 20, info_y), 1)
    info_y += 15

    # Contrôles
    ctrl_title = font.render("Contrôles :", True, YELLOW)
    screen.blit(ctrl_title, (info_x, info_y))
    info_y += 28

    controles = [
        "↑ ↓ ← → : Déplacer",
        "R        : Reset",
        "ESPACE   : Auto-play",
        "ESC      : Quitter",
    ]
    for ctrl in controles:
        c_text = font_small.render(ctrl, True, LIGHT_BLUE)
        screen.blit(c_text, (info_x, info_y))
        info_y += 22


def main():
    pygame.init()

    env = DroneRescueEnv(grid_size=15, num_victims=5, battery_max=200)
    obs, _ = env.reset(seed=42)

    grid_size = env.grid_size
    width  = MARGIN + grid_size * CELL_SIZE + MARGIN + INFO_WIDTH + MARGIN
    height = MARGIN + grid_size * CELL_SIZE + MARGIN

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone Secouriste — Jeu Vidéo IA")

    font       = pygame.font.SysFont("Arial", 16, bold=True)
    font_small = pygame.font.SysFont("Arial", 15)
    font_big   = pygame.font.SysFont("Arial", 28, bold=True)

    clock     = pygame.time.Clock()
    auto_play = False
    game_over = False
    message   = ""

    running = True
    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Reset
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    game_over = False
                    message   = ""
                    auto_play = False

                # Auto-play ON/OFF
                if event.key == pygame.K_SPACE:
                    auto_play = not auto_play

                # Contrôle manuel
                if not game_over:
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3

        # Auto-play : action aléatoire automatique
        if auto_play and not game_over:
            action = env.action_space.sample()

        # Appliquer l'action
        if action is not None and not game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                game_over = True
                if env.victims_rescued == env.num_victims:
                    message = "MISSION ACCOMPLIE !"
                elif env.battery <= 0:
                    message = "BATTERIE ÉPUISÉE !"
                else:
                    message = "TEMPS ÉCOULÉ !"

        # Dessin
        draw_env(screen, env, font, font_small)

        # Message de fin
        if game_over and message:
            color = GREEN if "ACCOMPLIE" in message else RED
            msg_surf = font_big.render(message, True, color)
            mx = (grid_size * CELL_SIZE) // 2 + MARGIN - msg_surf.get_width() // 2
            my = grid_size * CELL_SIZE // 2 + MARGIN - msg_surf.get_height() // 2
            bg = pygame.Surface(
                (msg_surf.get_width() + 20, msg_surf.get_height() + 10))
            bg.set_alpha(200)
            bg.fill(DARK_GRAY)
            screen.blit(bg, (mx - 10, my - 5))
            screen.blit(msg_surf, (mx, my))
            hint = font_small.render("Appuyez sur R pour rejouer",
                                     True, YELLOW)
            screen.blit(hint, (mx, my + 40))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()