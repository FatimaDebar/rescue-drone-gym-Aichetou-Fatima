import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# GÉNOME À RÈGLES DE PRIORITÉ — 6 gènes seulement
# Beaucoup plus simple qu'un réseau neuronal → converge facilement en 40 gén.
#
# Génome = [w_victim, w_base, w_recharge, bat_return_threshold,
#           bat_recharge_threshold, w_explore]
#
# w_victim          : priorité d'aller vers la victime la plus proche
# w_base            : priorité de rentrer à la base
# w_recharge        : priorité d'aller recharger
# bat_return_thr    : seuil batterie (0-1) pour décider de rentrer
# bat_recharge_thr  : seuil batterie (0-1) pour décider de recharger
# w_explore         : probabilité d'action aléatoire (exploration)
# ─────────────────────────────────────────────────────────────────────────────

GENOME_SIZE = 6


def random_genome(rng=None):
    if rng is None:
        rng = np.random.default_rng()

    genome = np.array([
        rng.uniform(0.5, 2.0),   # w_victim          : fort par défaut
        rng.uniform(0.5, 2.0),   # w_base            : fort par défaut
        rng.uniform(0.3, 1.5),   # w_recharge        : modéré
        rng.uniform(0.1, 0.5),   # bat_return_thr    : rentrer si < 10-50%
        rng.uniform(0.1, 0.4),   # bat_recharge_thr  : recharger si < 10-40%
        rng.uniform(0.0, 0.3),   # w_explore         : exploration faible
    ], dtype=np.float32)

    return genome


def init_population(pop_size, seed=None):
    rng = np.random.default_rng(seed)
    return [random_genome(rng) for _ in range(pop_size)]


def _manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _move_toward(drone_x, drone_y, tx, ty, env):
    """Retourne l'action qui rapproche le drone de (tx, ty)."""
    dx = tx - drone_x
    dy = ty - drone_y

    if abs(dx) >= abs(dy):
        preferred = [1 if dx > 0 else 0, 3 if dy > 0 else 2]
    else:
        preferred = [3 if dy > 0 else 2, 1 if dx > 0 else 0]

    all_actions = preferred + [a for a in [0, 1, 2, 3] if a not in preferred]

    for action in all_actions:
        if action == 0: nx, ny = drone_x - 1, drone_y
        elif action == 1: nx, ny = drone_x + 1, drone_y
        elif action == 2: nx, ny = drone_x, drone_y - 1
        else:             nx, ny = drone_x, drone_y + 1

        if (0 <= nx < env.grid_size and
                0 <= ny < env.grid_size and
                env.grid[nx, ny] != 1):
            return action

    return np.random.randint(0, 4)


def _find_recharge(env):
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            if env.grid[x, y] == 4:
                return (x, y)
    return None


def select_action(genome, observation, env=None, rng=None):
    """
    Sélectionne une action selon les règles de priorité encodées dans le génome.
    Si env=None, retourne une action aléatoire (fallback).
    """
    if env is None:
        return np.random.randint(0, 4)

    if rng is None:
        rng = np.random.default_rng()

    # Décodage du génome
    w_victim         = float(genome[0])
    w_base           = float(genome[1])
    w_recharge       = float(genome[2])
    bat_return_thr   = float(np.clip(genome[3], 0.05, 0.95))
    bat_recharge_thr = float(np.clip(genome[4], 0.05, 0.90))
    w_explore        = float(np.clip(genome[5], 0.0, 1.0))

    # Exploration aléatoire
    if rng.random() < w_explore:
        return int(rng.integers(0, 4))

    # Position drone
    drone_x, drone_y = env.drone_pos
    battery_pct = env.battery / env.battery_max
    base_x, base_y = env.base_pos
    dist_to_base = _manhattan(drone_x, drone_y, base_x, base_y)

    # Score de chaque stratégie
    scores = {}

    # 1. Rentrer à la base
    urgency_base = max(0.0, bat_return_thr - battery_pct) * 10.0
    scores['base'] = w_base * urgency_base

    # Retour obligatoire si batterie critique
    if battery_pct <= bat_return_thr or \
       (env.battery <= dist_to_base + 5 and env.victims_rescued > 0):
        return _move_toward(drone_x, drone_y, base_x, base_y, env)

    # Plus de victimes → rentrer
    if len(env.victims_positions) == 0:
        return _move_toward(drone_x, drone_y, base_x, base_y, env)

    # 2. Recharger si batterie basse
    recharge_pos = _find_recharge(env)
    if recharge_pos and battery_pct <= bat_recharge_thr:
        rx, ry = recharge_pos
        dist_recharge = _manhattan(drone_x, drone_y, rx, ry)
        urgency_recharge = max(0.0, bat_recharge_thr - battery_pct) * 8.0
        scores['recharge'] = w_recharge * urgency_recharge
        if scores.get('recharge', 0) > scores.get('base', 0):
            return _move_toward(drone_x, drone_y, rx, ry, env)

    # 3. Aller vers la victime la plus proche
    if env.victims_positions:
        dists = [(_manhattan(drone_x, drone_y, vx, vy), vx, vy)
                 for vx, vy in env.victims_positions]
        min_dist, vx, vy = min(dists)
        proximity_score = w_victim * (1.0 / (min_dist + 1.0)) * 10.0
        scores['victim'] = proximity_score

    # Choisir la meilleure stratégie
    best = max(scores, key=scores.get)

    if best == 'victim' and env.victims_positions:
        dists = [(_manhattan(drone_x, drone_y, vx, vy), vx, vy)
                 for vx, vy in env.victims_positions]
        _, vx, vy = min(dists)
        return _move_toward(drone_x, drone_y, vx, vy, env)
    elif best == 'recharge' and recharge_pos:
        return _move_toward(drone_x, drone_y, recharge_pos[0], recharge_pos[1], env)
    else:
        return _move_toward(drone_x, drone_y, base_x, base_y, env)


def evaluate_genome(genome, env, n_episodes=5, seed=0):
    """
    Évalue le génome sur plusieurs épisodes avec seeds variés.
    """
    total_fitness = 0.0
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        ep_seed = seed * 100 + ep * 13
        obs, _ = env.reset(seed=ep_seed)
        episode_reward = 0.0
        terminated = False
        truncated  = False

        while not terminated and not truncated:
            action = select_action(genome, obs, env=env, rng=rng)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        victims  = info["victims_rescued"]
        battery  = info["battery"]
        returned = terminated and battery > 0

        victims_bonus  = victims * 60.0
        return_bonus   = 40.0 if (returned and victims > 0) else 0.0
        complete_bonus = 100.0 if (victims == env.num_victims and returned) else 0.0
        bat_penalty    = -50.0 if battery <= 0 else 0.0

        total_fitness += (episode_reward + victims_bonus
                         + return_bonus + complete_bonus + bat_penalty)

    return total_fitness / n_episodes


def clone_genome(genome):
    return genome.copy()


def decode_genome(genome):
    """Retourne les paramètres lisibles du génome."""
    return {
        "w_victim":         round(float(genome[0]), 3),
        "w_base":           round(float(genome[1]), 3),
        "w_recharge":       round(float(genome[2]), 3),
        "bat_return_thr":   round(float(np.clip(genome[3], 0.05, 0.95)), 3),
        "bat_recharge_thr": round(float(np.clip(genome[4], 0.05, 0.90)), 3),
        "w_explore":        round(float(np.clip(genome[5], 0.0, 1.0)), 3),
    }


def genome_info():
    print("=" * 50)
    print("STRUCTURE DU GÉNOME (règles de priorité)")
    print("=" * 50)
    print(f"  GENOME_SIZE : {GENOME_SIZE} gènes")
    print(f"  [0] w_victim         : priorité victime")
    print(f"  [1] w_base           : priorité retour base")
    print(f"  [2] w_recharge       : priorité recharge")
    print(f"  [3] bat_return_thr   : seuil batterie retour")
    print(f"  [4] bat_recharge_thr : seuil batterie recharge")
    print(f"  [5] w_explore        : taux exploration")
    print("=" * 50)


if __name__ == "__main__":
    genome_info()
    g = random_genome(np.random.default_rng(42))
    print(f"\nGénome exemple : {g}")
    print(f"Décodé : {decode_genome(g)}")
    pop = init_population(pop_size=5, seed=42)
    print(f"\nPopulation de {len(pop)} individus créée.")