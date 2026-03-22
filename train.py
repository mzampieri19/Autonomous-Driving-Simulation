import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from env import CarEnv
from agent import Agent
from visualization import WorldVisualizer

"""
Training script for the Q-learning agent on the CarEnv environment.
- Phase 1: Train on a fixed world for PHASE1_EPISODES episodes.
- Phase 2: Continue training on random worlds for PHASE2_EPISODES episodes.
- Saves the trained agent and reward history.
- Generates and saves training curves for rewards and steps.
"""

AGENT_SAVE_PATH  = os.path.join("outputs", "saved_agent")
REWARD_HIST_PATH = os.path.join("outputs", "reward_history.npy")
PHASE1_END_PATH  = os.path.join("outputs", "phase1_end.npy")
CURVES_PATH      = os.path.join("outputs", "training_curves.png")

PHASE1_EPISODES  = 3000
PHASE2_EPISODES  = 2000
RENDER_EVERY     = 500
PRINT_EVERY      = 250
WINDOW           = 100
FORCE_RETRAIN    = True   # Set False to resume from saved_agent.pkl

ENV_KWARGS_BASE = dict(
    grid_size=20, num_branches=10, max_length=15,
    num_cars=60,  highway_max_cars=3, max_steps=500,
)

os.makedirs("outputs", exist_ok=True)

env_fixed  = CarEnv(**ENV_KWARGS_BASE, fixed_world=True)
env_random = CarEnv(**ENV_KWARGS_BASE, fixed_world=False)

if not FORCE_RETRAIN and os.path.exists(f"{AGENT_SAVE_PATH}.pkl"):
    print(f"Resuming from {AGENT_SAVE_PATH}.pkl ...")
    agent = Agent.from_file(AGENT_SAVE_PATH, env_fixed)
else:
    print("FORCE_RETRAIN=True — starting fresh." if FORCE_RETRAIN
          else "No saved agent — starting fresh.")
    agent = Agent(env_fixed, alpha=0.1, gamma=0.95,
                  epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998)

env_fixed.reset()
viz = WorldVisualizer(env_fixed.world, agent_pos=env_fixed.agent_pos)

reward_history  = []
steps_history   = []
rolling_rewards = deque(maxlen=WINDOW)

print(f"\n── Phase 1: Fixed world ({PHASE1_EPISODES} episodes) ──")
agent.env = env_fixed

# Fixed world training loop
for episode in range(1, PHASE1_EPISODES + 1):
    render    = (episode % RENDER_EVERY == 0)
    render_cb = (lambda e: viz.update(agent_pos=e.agent_pos)) if render else None
    total_reward, steps = agent.run_episode(render_callback=render_cb)

    reward_history.append(total_reward)
    steps_history.append(steps)
    rolling_rewards.append(total_reward)

    if episode % PRINT_EVERY == 0:
        print(f"[P1] Ep {episode:>5} | "
              f"Avg reward: {np.mean(rolling_rewards):>8.1f} | "
              f"Avg steps: {np.mean(steps_history[-WINDOW:]):>5.1f} | "
              f"ε: {agent.epsilon:.3f} | Q: {agent.q_table_size}")

# Random world training loop
print(f"\n── Phase 2: Random worlds ({PHASE2_EPISODES} episodes) ──")
agent.env     = env_random
agent.epsilon = max(agent.epsilon, 0.3)

for episode in range(1, PHASE2_EPISODES + 1):
    render    = (episode % RENDER_EVERY == 0)
    render_cb = (lambda e: viz.update(agent_pos=e.agent_pos)) if render else None
    total_reward, steps = agent.run_episode(render_callback=render_cb)

    reward_history.append(total_reward)
    steps_history.append(steps)
    rolling_rewards.append(total_reward)

    if episode % PRINT_EVERY == 0:
        print(f"[P2] Ep {episode:>5} | "
              f"Avg reward: {np.mean(rolling_rewards):>8.1f} | "
              f"Avg steps: {np.mean(steps_history[-WINDOW:]):>5.1f} | "
              f"ε: {agent.epsilon:.3f} | Q: {agent.q_table_size}")

# Save results and agent
np.save(REWARD_HIST_PATH, reward_history)
np.save(PHASE1_END_PATH,  PHASE1_EPISODES)
agent.save(AGENT_SAVE_PATH)

# Training curves
rewards = np.array(reward_history)
roll_r  = [np.mean(rewards[max(0, i-WINDOW):i+1]) for i in range(len(rewards))]
roll_s  = [np.mean(steps_history[max(0, i-WINDOW):i+1]) for i in range(len(steps_history))]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(rewards,  alpha=0.3, color="steelblue", label="Episode reward")
ax1.plot(roll_r,   color="steelblue", label=f"{WINDOW}-ep average")
ax1.axvline(PHASE1_EPISODES, color="grey", ls="--", lw=1, label="Phase 1/2 boundary")
ax1.set_ylabel("Total reward"); ax1.legend(); ax1.set_title("Training progress")

ax2.plot(steps_history, alpha=0.3, color="coral", label="Steps")
ax2.plot(roll_s, color="coral", label=f"{WINDOW}-ep average")
ax2.axvline(PHASE1_EPISODES, color="grey", ls="--", lw=1)
ax2.set_ylabel("Steps"); ax2.set_xlabel("Episode"); ax2.legend()

plt.tight_layout()
plt.savefig(CURVES_PATH, dpi=150)
print(f"Saved → {CURVES_PATH}")
plt.show()

# Final greedy episode visualization
print("\nRunning final greedy episode...")
agent.epsilon = 0.0
env_fixed.reset()
viz.world = env_fixed.world
total_reward, steps = agent.run_episode(
    render_callback=lambda e: viz.update(agent_pos=e.agent_pos)
)
print(f"Final episode — reward: {total_reward:.1f}  steps: {steps}")
viz.show()