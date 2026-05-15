import numpy as np

class RandomAgent:
    """
    Agent aléatoire — choisit une action au hasard à chaque step.
    Sert de baseline minimum de comparaison.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        # Ignore l'observation, choisit au hasard
        return self.action_space.sample()


def run_random_agent(env, num_episodes=30, seed=42):
    """
    Lance l'agent aléatoire sur plusieurs épisodes.
    Retourne les résultats pour analyse.
    """
    np.random.seed(seed)
    agent = RandomAgent(env.action_space)

    results = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0
        victims_rescued = 0
        returned_to_base = False
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            victims_rescued = info["victims_rescued"]

        # Vérifie si le drone est rentré à la base
        if terminated and info["battery"] > 0:
            returned_to_base = True

        results.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "victims_rescued": victims_rescued,
            "returned_to_base": returned_to_base,
            "steps": info["steps"]
        })

        print(f"[Aléatoire] Épisode {episode+1:2d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Victimes: {victims_rescued}/{env.num_victims} | "
              f"Retour: {'✅' if returned_to_base else '❌'} | "
              f"Steps: {info['steps']}")

    return results