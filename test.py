from environment.drone_rescue_env import DroneRescueEnv
from baselines.random_agent import run_random_agent
from baselines.heuristic_agent import run_heuristic_agent

env = DroneRescueEnv()

print("=" * 50)
print("🎲 AGENT ALÉATOIRE")
print("=" * 50)
run_random_agent(env, num_episodes=5)

print("\n" + "=" * 50)
print("🧠 AGENT HEURISTIQUE")
print("=" * 50)
run_heuristic_agent(env, num_episodes=5)