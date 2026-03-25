import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from env import CarEnv
from agent import Agent

PARAM_GRID = {
    "alpha":           [0.01, 0.05, 0.1,  0.3,  0.5  ],
    "gamma":           [0.80, 0.90, 0.95, 0.99, 0.999],
    "epsilon_decay":   [0.990, 0.995, 0.998, 0.999   ],
    "phase1_episodes": [500,  1500, 3000, 5000        ],
    "phase2_episodes": [500,  1000, 2000, 5000        ],
}

ENV_KWARGS = dict(
    grid_size=16, num_branches=8, max_length=5,
    num_cars=12,  highway_max_cars=3, max_steps=400,
)

EVAL_EPISODES   = 50
OUTPUT_DIR      = "outputs/optimization"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "optimization_checkpoint.csv")
RESULTS_CSV     = os.path.join(OUTPUT_DIR, "optimization_results.csv")
RESULTS_PNG     = os.path.join(OUTPUT_DIR, "optimization_results.png")
BEST_AGENT_PATH = os.path.join(OUTPUT_DIR, "best_agent")

def load_checkpoint():
    """
    Load completed trials from checkpoint CSV.
    Returns a dict mapping frozen param tuples --> score,
    and the list of already-completed result rows.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        return {}, []
    df   = pd.read_csv(CHECKPOINT_PATH)
    keys = list(PARAM_GRID.keys())
    done = {}
    rows = df.to_dict("records")
    for row in rows:
        key = tuple(row[k] for k in keys)
        done[key] = row["mean_reward"]
    print(f"Resuming: {len(done)} trials already completed.\n")
    return done, rows


def save_checkpoint(results):
    """Append latest results to checkpoint CSV."""
    df = pd.DataFrame(results)
    df.to_csv(CHECKPOINT_PATH, index=False)


def run_trial(alpha, gamma, epsilon_decay, phase1_episodes, phase2_episodes):
    """
    Train an agent with the given hyperparameters and return its
    mean reward over EVAL_EPISODES greedy episodes.
    """
    env_fixed  = CarEnv(**ENV_KWARGS, fixed_world=True)
    env_random = CarEnv(**ENV_KWARGS, fixed_world=False)

    agent = Agent(
        env_fixed,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
    )

    agent.env = env_fixed
    for _ in range(phase1_episodes):
        agent.run_episode()

    agent.env = env_random
    agent.epsilon = max(agent.epsilon, 0.3)
    for _ in range(phase2_episodes):
        agent.run_episode()

    agent.epsilon = 0.0
    rewards = []
    eval_env = CarEnv(**ENV_KWARGS, fixed_world=False)

    for _ in range(EVAL_EPISODES):
        raw_obs, _ = eval_env.reset()
        obs = tuple(raw_obs)
        total_reward = 0
        while True:
            was_in_highway = eval_env.in_highway
            action = agent.choose_action(obs)
            raw_next, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            obs = tuple(raw_next)
            if terminated or truncated:
                break
        rewards.append(total_reward)

    return float(np.mean(rewards)), agent

def grid_search():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    # Load any previously completed trials
    done_map, results = load_checkpoint()
    remaining = [c for c in combos if tuple(c) not in done_map]
    skipped   = total - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already-completed trials.\n")

    best_score  = max((r["mean_reward"] for r in results), default=-np.inf)
    best_params = None
    best_agent  = None

    # Find best params from checkpoint if resuming
    if results:
        best_row = max(results, key=lambda r: r["mean_reward"])
        best_params = {k: best_row[k] for k in keys}

    trial_num = skipped  # Continue numbering from where we left off

    try:
        for combo in remaining:
            params = dict(zip(keys, combo))
            trial_num += 1
            t0 = time.time()

            score, agent = run_trial(**params)

            elapsed = time.time() - t0

            if score > best_score:
                best_score  = score
                best_agent  = agent
                best_params = params.copy()
                marker = " ◄ NEW BEST"
            else:
                marker = ""

            row = {**params, "mean_reward": score, "trial": trial_num}
            results.append(row)

            # Checkpoint after every trial
            save_checkpoint(results)

            print(
                f"[{trial_num:>4}/{total}] "
                + " | ".join(f"{k}={v}" for k, v in params.items())
                + f"  →  {score:>8.2f}  ({elapsed:.1f}s){marker}"
            )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at trial {trial_num}/{total}.")
        print(f"Progress saved to {CHECKPOINT_PATH} — rerun to continue.")

    df = pd.DataFrame(results).sort_values("mean_reward", ascending=False)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll results saved → {RESULTS_CSV}")

    if best_agent is not None:
        best_agent.save(BEST_AGENT_PATH)
        print(f"Best agent saved → {BEST_AGENT_PATH}.pkl")
    else:
        print("No new best agent found in this run (all trials resumed from checkpoint).")

    print("\n── Top 10 Configurations ──────────────────────────────────")
    print(df.head(10).to_string(index=False))

    print("\n── Best Configuration ─────────────────────────────────────")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k:<20} {v}")
        print(f"  {'mean_reward':<20} {best_score:.2f}")

    return df, best_params, best_score

def plot_results(df):
    """
    6-panel figure showing how each hyperparameter affects mean reward.
    Each panel shows the marginal effect of one parameter averaged
    over all other parameter combinations.
    """
    params = list(PARAM_GRID.keys())
    n_params = len(params)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Hyperparameter Grid Search  —  {len(df)} trials  |  "
        f"5×5×4×4×4 = 1,600 combinations  |  "
        f"Metric: mean reward over {EVAL_EPISODES} eval episodes",
        fontsize=13, fontweight="bold",
    )

    # Use a 2×3 grid: 5 parameter panels + 1 top-10 table
    gs_layout = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = plt.cm.tab10.colors

    for idx, param in enumerate(params):
        ax  = fig.add_subplot(gs_layout[idx // 3, idx % 3])
        grouped = df.groupby(param)["mean_reward"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        bars = ax.bar(
            [str(v) for v in means.index],
            means.values,
            yerr=stds.values,
            capsize=5,
            color=colors[idx],
            edgecolor="black",
            alpha=0.85,
            error_kw={"elinewidth": 1.5},
        )

        # Annotate bar tops
        for bar, mean_val in zip(bars, means.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + stds.values.max() * 0.05,
                f"{mean_val:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        ax.set_title(param, fontweight="bold", fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Avg reward (±std)", fontsize=8)
        ax.tick_params(labelsize=8)

        # Highlight the best value
        best_val = means.idxmax()
        for bar, val in zip(bars, means.index):
            if val == best_val:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.5)

    # Panel 6: top 10 table 
    ax_table = fig.add_subplot(gs_layout[1, 2])
    ax_table.axis("off")

    top10   = df.head(10).reset_index(drop=True)
    col_labels = ["#", "α", "γ", "ε_decay", "P1", "P2", "Reward"]
    table_data = [
        [
            str(i + 1),
            str(row["alpha"]),
            str(row["gamma"]),
            str(row["epsilon_decay"]),
            str(int(row["phase1_episodes"])),
            str(int(row["phase2_episodes"])),
            f"{row['mean_reward']:.1f}",
        ]
        for i, row in top10.iterrows()
    ]

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # Highlight header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1976D2")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best row
    for j in range(len(col_labels)):
        tbl[1, j].set_facecolor("#C8E6C9")

    ax_table.set_title("Top 10 Configurations", fontweight="bold", fontsize=10)

    plt.savefig(RESULTS_PNG, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {RESULTS_PNG}")
    plt.show()

if __name__ == "__main__":
    df, best_params, best_score = grid_search()
    plot_results(df)

    print("\n── Suggested train.py settings ────────────────────────────")
    print("Update these values in train.py to use the best configuration:\n")
    print(f"  agent = Agent(env_fixed,")
    print(f"      alpha         = {best_params['alpha']},")
    print(f"      gamma         = {best_params['gamma']},")
    print(f"      epsilon       = 1.0,")
    print(f"      epsilon_min   = 0.01,")
    print(f"      epsilon_decay = {best_params['epsilon_decay']},")
    print(f"  )")
    print(f"  PHASE1_EPISODES = {int(best_params['phase1_episodes'])}")
    print(f"  PHASE2_EPISODES = {int(best_params['phase2_episodes'])}")