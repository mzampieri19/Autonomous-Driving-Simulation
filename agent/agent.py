import os
import pickle
import random
import numpy as np


class Agent:
    """
    Tabular Q-Learning agent that operates on a CarEnv (Gymnasium interface).

    State:  (grid_x, grid_y, in_highway, hw_row, hw_col, orientation)
    Q-table: dict { (state_tuple, action): float }
    """

    def __init__(self, env, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """Initialize the agent with learning parameters and an empty Q-table."""
        self.env           = env
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions     = env.action_space.n
        self.q_table: dict = {}
        self._obs          = None

    def _q(self, obs, action):
        """Get the Q-value for a given state and action, defaulting to 0.0 if not present."""
        return self.q_table.get((obs, action), 0.0)

    def _best_action(self, obs):
        """Return the action with the highest Q-value for the given state."""
        return max(range(self.n_actions), key=lambda a: self._q(obs, a))

    def _update(self, obs, action, reward, next_obs, terminated):
        """Update the Q-table based on the observed transition and reward."""
        best_next = 0.0 if terminated else max(
            self._q(next_obs, a) for a in range(self.n_actions)
        )
        old_q = self._q(obs, action)
        self.q_table[(obs, action)] = old_q + self.alpha * (
            reward + self.gamma * best_next - old_q
        )

    def choose_action(self, obs):
        """Choose an action using epsilon-greedy strategy based on the current observation."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self._best_action(obs)

    def run_episode(self, render_callback=None):
        """
        Run one full episode. Returns (total_reward, steps).
        render_callback(env) is called after each step if provided.
        """
        raw_obs, _ = self.env.reset()
        obs = tuple(raw_obs)
        total_reward = 0
        steps = 0

        while True:
            if render_callback:
                render_callback(self.env)
            action = self.choose_action(obs)
            raw_next, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = tuple(raw_next)
            self._update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        self.decay_epsilon()
        return total_reward, steps

    def decay_epsilon(self):
        """Decay the exploration rate epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def q_table_size(self):
        """Return the number of state-action pairs currently in the Q-table."""
        return len(self.q_table)

    def save(self, path: str):
        """Save the agent's Q-table and parameters to a file."""
        if not path.endswith(".pkl"):
            path += ".pkl"
        payload = {
            "q_table":       self.q_table,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Agent saved → {path}  ({self.q_table_size} states)")

    def load(self, path: str):
        """Load the agent's Q-table and parameters from a file."""
        if not path.endswith(".pkl"):
            path += ".pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved agent at '{path}'")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q_table       = payload["q_table"]
        self.epsilon       = payload["epsilon"]
        self.epsilon_min   = payload["epsilon_min"]
        self.epsilon_decay = payload["epsilon_decay"]
        self.alpha         = payload["alpha"]
        self.gamma         = payload["gamma"]
        print(f"Agent loaded ← {path}  ({self.q_table_size} states)")

    @classmethod
    def from_file(cls, path: str, env):
        """Create an agent instance and load its state from a file."""
        agent = cls(env)
        agent.load(path)
        return agent