import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from env import CarEnv
from agent import Agent

OUTPUT_DIR      = "outputs/evaluation"
AGENT_PATH      = "outputs/training/saved_agent"
OUTPUT_PATH     = os.path.join(OUTPUT_DIR, "evaluation.png")
EVAL_EPISODES   = 500
RANDOM_EPISODES = 500

ENV_KWARGS = dict(
    grid_size=16, num_branches=8, max_length=5,
    num_cars=12, highway_max_cars=3, max_steps=400,
)

def make_env(**kwargs):
    return CarEnv(**ENV_KWARGS, **kwargs)

def run_episodes(agent, env, n, epsilon=0.0):
    agent.epsilon = epsilon
    stats = defaultdict(list)
    for _ in range(n):
        obs, _ = env.reset()
        obs    = tuple(obs)
        total_reward = steps = highway_steps = car_hits = 0
        reached_goal = False
        while True:
            # Capture highway state BEFORE the step so exit steps are counted
            was_in_highway = env.in_highway
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if was_in_highway:
                highway_steps += 1
            if reward == env.R_HIT_CAR:
                car_hits += 1
            total_reward += reward
            steps += 1
            obs = tuple(next_obs)
            if terminated:
                reached_goal = True
                break
            if truncated:
                break
        stats["reward"].append(total_reward)
        stats["steps"].append(steps)
        stats["reached_goal"].append(int(reached_goal))
        stats["highway_steps"].append(highway_steps)
        stats["highway_ratio"].append(highway_steps / max(steps, 1))
        stats["car_hits"].append(car_hits)
    return {k: np.array(v) for k, v in stats.items()}

def random_agent_stats(env, n):
    stats = defaultdict(list)
    for _ in range(n):
        env.reset()
        total_reward = steps = 0
        reached_goal = False
        for _ in range(ENV_KWARGS["max_steps"]):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total_reward += reward
            steps += 1
            if terminated:
                reached_goal = True
                break
            if truncated:
                break
        stats["reward"].append(total_reward)
        stats["steps"].append(steps)
        stats["reached_goal"].append(int(reached_goal))
    return {k: np.array(v) for k, v in stats.items()}

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def fmt(label, value, unit=""):
    print(f"  {label:<35} {value}{unit}")

env   = make_env()
agent = Agent.from_file(AGENT_PATH, env)

print("\nRunning trained agent evaluation ...")
trained  = run_episodes(agent, env, EVAL_EPISODES)
print("Running random baseline ...")
random_s = random_agent_stats(make_env(), RANDOM_EPISODES)

success_mask = trained["reached_goal"] == 1

section("AGENT EVALUATION REPORT")
fmt("Episodes evaluated",  EVAL_EPISODES)
fmt("Grid size",           f"{ENV_KWARGS['grid_size']}×{ENV_KWARGS['grid_size']}")
fmt("Max steps/episode",   ENV_KWARGS["max_steps"])
fmt("Q-table states",      agent.q_table_size)

section("1. Goal Completion")
fmt("Success rate (trained)", f"{trained['reached_goal'].mean()*100:.1f}", "%")
fmt("Success rate (random)",  f"{random_s['reached_goal'].mean()*100:.1f}", "%")

section("2. Reward")
fmt("Mean reward (trained)", f"{trained['reward'].mean():.1f}")
fmt("Std  reward (trained)", f"{trained['reward'].std():.1f}")
fmt("Min  reward (trained)", f"{trained['reward'].min():.1f}")
fmt("Max  reward (trained)", f"{trained['reward'].max():.1f}")
fmt("Mean reward (random)",  f"{random_s['reward'].mean():.1f}")
fmt("Improvement over random",
    f"{trained['reward'].mean() - random_s['reward'].mean():.1f}", " pts")

section("3. Efficiency (successful episodes only)")
if success_mask.any():
    fmt("Mean steps to goal",   f"{trained['steps'][success_mask].mean():.1f}")
    fmt("Median steps to goal", f"{np.median(trained['steps'][success_mask]):.1f}")
    fmt("Best steps to goal",   f"{trained['steps'][success_mask].min()}")
else:
    print("  No successful episodes recorded.")

section("4. Highway Usage")
fmt("Mean highway step ratio", f"{trained['highway_ratio'].mean()*100:.1f}", "%")
fmt("Mean highway steps/ep",   f"{trained['highway_steps'].mean():.1f}")

section("5. Safety")
fmt("Mean car hits/episode",   f"{trained['car_hits'].mean():.3f}")
fmt("Episodes with 0 car hits",f"{(trained['car_hits']==0).mean()*100:.1f}", "%")
fmt("Max car hits in one ep",  f"{trained['car_hits'].max()}")

c_t, c_r, c_a = "#2196F3", "#FF7043", "#4CAF50"

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    f"Evaluation Report  —  {EVAL_EPISODES} episodes  |  "
    f"Grid {ENV_KWARGS['grid_size']}×{ENV_KWARGS['grid_size']}  |  "
    f"Q-table: {agent.q_table_size} states",
    fontsize=13, fontweight="bold",
)

# 1. Reward distribution
ax = axes[0, 0]
bins = np.linspace(min(trained["reward"].min(), random_s["reward"].min()),
                   max(trained["reward"].max(), random_s["reward"].max()), 35)
ax.hist(random_s["reward"], bins=bins, alpha=0.6, color=c_r,
        label=f"Random  (μ={random_s['reward'].mean():.0f})")
ax.hist(trained["reward"],  bins=bins, alpha=0.7, color=c_t,
        label=f"Trained (μ={trained['reward'].mean():.0f})")
ax.axvline(trained["reward"].mean(),  color=c_t, lw=2, ls="--")
ax.axvline(random_s["reward"].mean(), color=c_r, lw=2, ls="--")
ax.set_title("Reward Distribution"); ax.set_xlabel("Total reward")
ax.set_ylabel("Episodes"); ax.legend()

# 2. Success rate
ax = axes[0, 1]
rates = [random_s["reached_goal"].mean()*100, trained["reached_goal"].mean()*100]
bars  = ax.bar(["Random", "Trained"], rates, color=[c_r, c_t],
               width=0.45, edgecolor="black")
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
            f"{rate:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 110); ax.set_title("Goal Success Rate")
ax.set_ylabel("% episodes reaching goal")

# 3. Steps to goal
ax = axes[0, 2]
if success_mask.any():
    sd = trained["steps"][success_mask]
    ax.hist(sd, bins=20, color=c_a, edgecolor="black", alpha=0.85)
    ax.axvline(sd.mean(),      color="black", lw=2, ls="--",
               label=f"Mean {sd.mean():.1f}")
    ax.axvline(np.median(sd),  color="grey",  lw=1.5, ls=":",
               label=f"Median {np.median(sd):.1f}")
    ax.set_title("Steps to Goal (successful eps)")
    ax.set_xlabel("Steps"); ax.set_ylabel("Episodes"); ax.legend()
else:
    ax.text(0.5, 0.5, "No successful\nepisodes",
            ha="center", va="center", transform=ax.transAxes, fontsize=13)
    ax.set_title("Steps to Goal")

# 4. Highway usage
ax = axes[1, 0]
ax.scatter(range(len(trained["highway_ratio"])),
           trained["highway_ratio"]*100, s=4, alpha=0.4, color=c_t)
w = 30
if len(trained["highway_ratio"]) >= w:
    roll = np.convolve(trained["highway_ratio"]*100, np.ones(w)/w, mode="valid")
    ax.plot(range(w-1, len(trained["highway_ratio"])), roll,
            color="black", lw=2, label=f"{w}-ep avg")
ax.set_title("Highway Usage per Episode")
ax.set_xlabel("Episode"); ax.set_ylabel("% steps on highway"); ax.legend()

# 5. Car collisions
ax = axes[1, 1]
max_h = int(trained["car_hits"].max()) + 1
ax.bar(range(max_h),
       [int((trained["car_hits"]==h).sum()) for h in range(max_h)],
       color="#EF5350", edgecolor="black")
ax.set_title("Car Collisions per Episode")
ax.set_xlabel("Collisions"); ax.set_ylabel("Episodes")
ax.set_xticks(range(max_h))

# 6. Scorecard
ax = axes[1, 2]
ax.axis("off")
rows = [
    ("Metric",               "Trained",                                              "Random"),
    ("─"*18,                 "─"*10,                                                "─"*10),
    ("Success rate",         f"{trained['reached_goal'].mean()*100:.1f}%",          f"{random_s['reached_goal'].mean()*100:.1f}%"),
    ("Mean reward",          f"{trained['reward'].mean():.1f}",                     f"{random_s['reward'].mean():.1f}"),
    ("Std reward",           f"{trained['reward'].std():.1f}",                      f"{random_s['reward'].std():.1f}"),
    ("Avg steps (wins)",     f"{trained['steps'][success_mask].mean():.1f}" if success_mask.any() else "N/A", "N/A"),
    ("Avg highway %",        f"{trained['highway_ratio'].mean()*100:.1f}%",         "N/A"),
    ("Avg car hits/ep",      f"{trained['car_hits'].mean():.3f}",                   "N/A"),
    ("Clean eps (0 hits)",   f"{(trained['car_hits']==0).mean()*100:.1f}%",         "N/A"),
    ("Q-table states",       f"{agent.q_table_size}",                               "N/A"),
]
for i, row in enumerate(rows):
    for j, val in enumerate([0.0, 0.55, 0.80]):
        ax.text(val, 1 - i*0.095, row[j], transform=ax.transAxes,
                fontsize=10, fontweight="bold" if i < 2 else "normal",
                va="top", fontfamily="monospace")
ax.set_title("Summary Scorecard", fontweight="bold")

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUTPUT_PATH}")
plt.show()