import numpy as np

class HeuristicAgent:

    def __init__(self):
        self.stuck_counter = 0
        self.last_pos = None
        self.random_escape = 0
        self.recharge_used = False

    def select_action(self, observation, env):
        drone_x = int(observation[0])
        drone_y = int(observation[1])
        battery  = int(observation[2])
        victims_left = int(observation[3])

        base_x, base_y = env.base_pos
        dist_to_base = abs(drone_x - base_x) + abs(drone_y - base_y)

        #Trouver la station de recharge
        recharge_pos = self._find_recharge(env)

        # Détection blocage 
        current_pos = (drone_x, drone_y)
        if current_pos == self.last_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_pos = current_pos

        if self.stuck_counter >= 3:
            self.stuck_counter = 0
            self.random_escape = 5

        if self.random_escape > 0:
            self.random_escape -= 1
            return self._random_free_action(drone_x, drone_y, env)

        # Aller recharger si batterie moyenne et recharge proche
        if recharge_pos and not self.recharge_used:
            rx, ry = recharge_pos
            dist_to_recharge = abs(drone_x - rx) + abs(drone_y - ry)
            battery_pct = battery / env.battery_max

            # Si batterie entre 20% et 40% ET recharge plus proche que la base
            if 0.20 <= battery_pct <= 0.40 and dist_to_recharge < dist_to_base:
                self.recharge_used = True
                return self._move_toward(drone_x, drone_y, rx, ry, env)

        # Reset recharge si batterie pleine
        if battery >= env.battery_max * 0.9:
            self.recharge_used = False

        # Rentrer si batterie insuffisante 
        if battery <= dist_to_base + 8:
            return self._move_toward(drone_x, drone_y, base_x, base_y, env)

        # Plus de victimes -> rentre à la base
        if victims_left == 0:
            return self._move_toward(drone_x, drone_y, base_x, base_y, env)

        # Va vers la victime la plus proche
        if len(env.victims_positions) > 0:
            closest = self._find_closest_victim(drone_x, drone_y, env.victims_positions)
            return self._move_toward(drone_x, drone_y, closest[0], closest[1], env)

        return self._move_toward(drone_x, drone_y, base_x, base_y, env)

    def _find_recharge(self, env):
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                if env.grid[x, y] == 4:
                    return (x, y)
        return None

    def _find_closest_victim(self, x, y, victims):
        closest = None
        min_dist = float('inf')
        for vx, vy in victims:
            dist = abs(vx - x) + abs(vy - y)
            if dist < min_dist:
                min_dist = dist
                closest = (vx, vy)
        return closest

    def _move_toward(self, x, y, tx, ty, env):
        dx = tx - x
        dy = ty - y

        if abs(dx) >= abs(dy):
            preferred = [1 if dx > 0 else 0, 3 if dy > 0 else 2]
        else:
            preferred = [3 if dy > 0 else 2, 1 if dx > 0 else 0]

        all_actions = preferred + [a for a in [0, 1, 2, 3] if a not in preferred]

        for action in all_actions:
            nx, ny = self._next_pos(x, y, action)
            if self._is_free(nx, ny, env):
                return action

        return self._random_free_action(x, y, env)

    def _random_free_action(self, x, y, env):
        free_actions = []
        for action in [0, 1, 2, 3]:
            nx, ny = self._next_pos(x, y, action)
            if self._is_free(nx, ny, env):
                free_actions.append(action)
        if free_actions:
            return np.random.choice(free_actions)
        return 0

    def _next_pos(self, x, y, action):
        if action == 0: return x - 1, y
        if action == 1: return x + 1, y
        if action == 2: return x, y - 1
        if action == 3: return x, y + 1
        return x, y

    def _is_free(self, x, y, env):
        return (0 <= x < env.grid_size and
                0 <= y < env.grid_size and
                env.grid[x, y] != 1)


def run_heuristic_agent(env, num_episodes=30, seed=42):
    np.random.seed(seed)
    results = []

    for episode in range(num_episodes):
        agent = HeuristicAgent()
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0
        victims_rescued = 0
        returned_to_base = False
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = agent.select_action(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            victims_rescued = info["victims_rescued"]

        if terminated and info["battery"] > 0:
            returned_to_base = True

        results.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "victims_rescued": victims_rescued,
            "returned_to_base": returned_to_base,
            "steps": info["steps"]
        })

        print(f"[Heuristique] Épisode {episode+1:2d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Victimes: {victims_rescued}/{env.num_victims} | "
              f"Retour: { 'ok' if returned_to_base else 'Non'} | "
              f"Steps: {info['steps']}")

    return results