import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneRescueEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=15, num_victims=5, battery_max=200):
        super().__init__()
        self.grid_size = grid_size
        self.num_victims = num_victims
        self.battery_max = battery_max
        self.action_space = spaces.Discrete(4)
        obs_size = 4 + 25
        self.observation_space = spaces.Box(
            low=0,
            high=max(grid_size, battery_max, num_victims),
            shape=(obs_size,),
            dtype=np.float32
        )
        self.grid = None
        self.drone_pos = None
        self.battery = None
        self.victims_rescued = 0
        self.victims_positions = []
        self.base_pos = None
        self.steps = 0
        self.max_steps = 300

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.steps = 0
        self.victims_rescued = 0

        num_obstacles = int(self.grid_size * self.grid_size * 0.15)
        obstacle_positions = self._random_positions(num_obstacles, exclude=[])
        for pos in obstacle_positions:
            self.grid[pos[0], pos[1]] = 1

        self.base_pos = (self.grid_size - 1, self.grid_size - 1)
        self.grid[self.base_pos[0], self.base_pos[1]] = 3

        recharge_pos = self._random_positions(1, exclude=obstacle_positions + [self.base_pos])
        self.grid[recharge_pos[0][0], recharge_pos[0][1]] = 4

        occupied = obstacle_positions + [self.base_pos] + recharge_pos
        self.victims_positions = self._random_positions(self.num_victims, exclude=occupied)
        for pos in self.victims_positions:
            self.grid[pos[0], pos[1]] = 2

        self.drone_pos = [0, 0]
        self.grid[0, 0] = 5
        self.battery = self.battery_max

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.steps += 1
        reward = -1
        terminated = False
        truncated = False

        x, y = self.drone_pos
        if action == 0:
            nx, ny = x - 1, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x, y - 1
        else:
            nx, ny = x, y + 1

        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            if self.grid[nx, ny] != 1:
                self.grid[x, y] = 0
                self.drone_pos = [nx, ny]
                cell = self.grid[nx, ny]

                if cell == 2:
                    reward += 20
                    self.victims_rescued += 1
                    self.victims_positions = [
                        p for p in self.victims_positions
                        if not (p[0] == nx and p[1] == ny)
                    ]
                elif cell == 4:
                    self.battery = min(self.battery_max, self.battery + 20)
                elif cell == 3:
                    reward += 10
                    if self.victims_rescued == self.num_victims:
                        reward += 50
                    terminated = True

                self.grid[nx, ny] = 5

        self.battery -= 1
        if self.battery <= 0:
            reward -= 20
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            "victims_rescued": self.victims_rescued,
            "battery": self.battery,
            "steps": self.steps
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        x, y = self.drone_pos
        vision = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    vision.append(float(self.grid[nx, ny]))
                else:
                    vision.append(-1.0)
        obs = np.array([
            float(x),
            float(y),
            float(self.battery),
            float(self.num_victims - self.victims_rescued)
        ] + vision, dtype=np.float32)
        return obs

    def render(self):
        symbols = {0: ".", 1: "#", 2: "V", 3: "B", 4: "R", 5: "D"}
        print(f"\n Batterie: {self.battery} | Victimes restantes: "
              f"{self.num_victims - self.victims_rescued} | Steps: {self.steps}")
        print("+" + "-" * (self.grid_size * 2 - 1) + "+")
        for row in self.grid:
            print("|" + " ".join(symbols.get(c, "?") for c in row) + "|")
        print("+" + "-" * (self.grid_size * 2 - 1) + "+")
        print("Légende: D=Drone  V=Victime  B=Base  R=Recharge  #=Obstacle  .=Vide")

    def _random_positions(self, n, exclude):
        positions = []
        while len(positions) < n:
            pos = (
                np.random.randint(1, self.grid_size - 1),
                np.random.randint(1, self.grid_size - 1)
            )
            if pos not in exclude and pos not in positions:
                positions.append(pos)
        return positions