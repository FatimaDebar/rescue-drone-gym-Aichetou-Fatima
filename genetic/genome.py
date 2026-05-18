import numpy as np

OBS_SIZE    = 29   
HIDDEN_SIZE = 16
ACTION_SIZE = 4

GENOME_SIZE = (HIDDEN_SIZE * OBS_SIZE) + HIDDEN_SIZE \
            + (ACTION_SIZE * HIDDEN_SIZE) + ACTION_SIZE

# Initialisation


def random_genome(rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # Xavier : écart-type = sqrt(2 / (fan_in + fan_out))
    std_w1 = np.sqrt(2.0 / (OBS_SIZE + HIDDEN_SIZE))
    std_w2 = np.sqrt(2.0 / (HIDDEN_SIZE + ACTION_SIZE))

    w1 = rng.normal(0, std_w1, HIDDEN_SIZE * OBS_SIZE).astype(np.float32)
    b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    w2 = rng.normal(0, std_w2, ACTION_SIZE * HIDDEN_SIZE).astype(np.float32)
    b2 = np.zeros(ACTION_SIZE, dtype=np.float32)

    return np.concatenate([w1, b1, w2, b2])


def init_population(pop_size, seed=None):
    rng = np.random.default_rng(seed)
    return [random_genome(rng) for _ in range(pop_size)]

# Décodage : génome → action

def decode_genome(genome):
    idx = 0

    w1_size = HIDDEN_SIZE * OBS_SIZE
    W1 = genome[idx: idx + w1_size].reshape(HIDDEN_SIZE, OBS_SIZE)
    idx += w1_size

    b1 = genome[idx: idx + HIDDEN_SIZE]
    idx += HIDDEN_SIZE

    w2_size = ACTION_SIZE * HIDDEN_SIZE
    W2 = genome[idx: idx + w2_size].reshape(ACTION_SIZE, HIDDEN_SIZE)
    idx += w2_size

    b2 = genome[idx: idx + ACTION_SIZE]

    return W1, b1, W2, b2


def select_action(genome, observation):
    W1, b1, W2, b2 = decode_genome(genome)

    # Normalisation légère de l'observation 
    obs = np.array(observation, dtype=np.float32)
    obs = obs / (np.linalg.norm(obs) + 1e-8)

    # Couche cachée avec activation ReLU
    h = np.maximum(0.0, W1 @ obs + b1)

    # Couche de sortie
    logits = W2 @ h + b2

    return int(np.argmax(logits))

# Évaluation fitness


def evaluate_genome(genome, env, n_episodes=3, seed=0):
    total_fitness = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = select_action(genome, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        # Bonus victimes sauvées (5 pts chacune si pas déjà dans reward)
        victims_bonus = info["victims_rescued"] * 2.0

        # Bonus retour en base
        return_bonus = 15.0 if (terminated and info["battery"] > 0) else 0.0

        total_fitness += episode_reward + victims_bonus + return_bonus

    return total_fitness / n_episodes

# Utilitaires

def clone_genome(genome):
    return genome.copy()


def genome_info():
    print("=" * 50)
    print("STRUCTURE DU GÉNOME")
    print("=" * 50)
    print(f"  Observation size : {OBS_SIZE}")
    print(f"  Hidden size      : {HIDDEN_SIZE}")
    print(f"  Action size      : {ACTION_SIZE}")
    print(f"  W1  ({HIDDEN_SIZE}×{OBS_SIZE}) = {HIDDEN_SIZE * OBS_SIZE} poids")
    print(f"  b1  ({HIDDEN_SIZE},)    = {HIDDEN_SIZE} biais")
    print(f"  W2  ({ACTION_SIZE}×{HIDDEN_SIZE})  = {ACTION_SIZE * HIDDEN_SIZE} poids")
    print(f"  b2  ({ACTION_SIZE},)     = {ACTION_SIZE} biais")
    print(f"  TOTAL            : {GENOME_SIZE} gènes")
    print("=" * 50)

# Test rapide (si exécuté directement)

if __name__ == "__main__":
    genome_info()

    # Création d'un génome aléatoire
    g = random_genome(np.random.default_rng(42))
    print(f"\nGénome créé : {g.shape}, dtype={g.dtype}")
    print(f"Min={g.min():.4f}  Max={g.max():.4f}  Moyenne={g.mean():.4f}")

    # Test de sélection d'action sur une obs fictive
    fake_obs = np.random.rand(OBS_SIZE).astype(np.float32)
    action = select_action(g, fake_obs)
    print(f"\nAction sélectionnée pour obs aléatoire : {action}  ({['↑','↓','←','→'][action]})")

    # Test de la population
    pop = init_population(pop_size=10, seed=42)
    print(f"\nPopulation de {len(pop)} individus créée.")
    print(f"Taille d'un individu : {pop[0].shape}")