import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .world import World


class CarEnv(gym.Env):
    """
    Gymnasium environment for the self-driving car simulation.

    Observation (6 integers):
        [grid_x, grid_y, in_highway, hw_row, hw_col, orientation]
        orientation: 0=none, 1=horizontal, 2=vertical

    Actions (7 discrete):
        0=Up  1=Down  2=Left  3=Right  4=Enter  5=Exit  6=Diagonal
    """

    metadata = {"render_modes": ["human"]}

    # Rewards 
    R_GOAL         =  200
    R_HIGHWAY_STEP =    5
    R_DIAGONAL     =    4
    R_STEP         =   -1
    R_INVALID      =   -1
    R_HIT_CAR      =  -40

    N_ACTIONS = 7

    def __init__(self, grid_size=8, num_branches=4, max_length=5,
                 num_cars=8, highway_max_cars=3, max_steps=200,
                 fixed_world=False):
        """Initialize the CarEnv with specified parameters for grid size, highway configuration, car placement, and episode length."""
        super().__init__()
        self.grid_size        = grid_size
        self.num_branches     = num_branches
        self.max_length       = max_length
        self.num_cars         = num_cars
        self.highway_max_cars = highway_max_cars
        self.max_steps        = max_steps
        self.fixed_world      = fixed_world

        self.world      = None
        self.grid_x     = 0
        self.grid_y     = 0
        self.in_highway = False
        self.hw_row     = 0
        self.hw_col     = 0
        self._steps     = 0
        self._visited   = {}

        self.action_space = spaces.Discrete(self.N_ACTIONS) # Action space
        self.observation_space = spaces.MultiDiscrete(
            [grid_size, grid_size, 2, 3, 3, 3] # Observation space dimensions
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state, creating a new world if not fixed, and return the initial observation."""
        super().reset(seed=seed)
        if not self.fixed_world or self.world is None:
            self.world = World(
                self.grid_size, self.num_branches, self.max_length,
                num_cars=self.num_cars, highway_max_cars=self.highway_max_cars,
            )
        self.grid_x, self.grid_y = self.world.start
        self.in_highway = False
        self.hw_row     = 0
        self.hw_col     = 0
        self._steps     = 0
        self._visited   = {}
        return self._obs(), {}

    def step(self, action):
        """Apply the given action, update the environment state, calculate the reward, and determine if the episode is terminated or truncated."""
        self._steps += 1
        reward, terminated = self._apply_action(int(action))

        s = tuple(self._obs())
        self._visited[s] = self._visited.get(s, 0) + 1
        if self._visited[s] > 3:
            reward += -2 * (self._visited[s] - 3)

        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        """Construct the observation array based on the current state of the agent in the grid and highway."""
        orientation = 0
        if self.in_highway:
            hw = self.world.grid[self.grid_x][self.grid_y].highway
            if hw:
                orientation = 1 if hw.orientation == "horizontal" else 2
        return np.array([
            self.grid_x,
            self.grid_y,
            int(self.in_highway),
            self.hw_row if self.in_highway else 0,
            self.hw_col if self.in_highway else 0,
            orientation,
        ], dtype=np.int64)

    def _apply_action(self, action):
        """Apply the given action based on whether the agent is currently in a highway cell or not, and return the resulting reward and termination status."""
        return self._highway_action(action) if self.in_highway \
               else self._grid_action(action)

    def _grid_action(self, action):
        """Apply a grid movement action or attempt to enter a highway, and return the resulting reward and termination status."""
        if action == 4:
            return self._enter_highway()
        if action in (5, 6):
            return self.R_INVALID, False
        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
        nx, ny = self.grid_x + dx, self.grid_y + dy
        if not self._in_bounds(nx, ny):
            return self.R_INVALID, False
        self.grid_x, self.grid_y = nx, ny
        if (nx, ny) == self.world.goal:
            return self.R_GOAL, True
        return self.R_STEP, False

    def _enter_highway(self):
        """Attempt to enter a highway cell, initializing highway position and checking for cars, and return the resulting reward and termination status."""
        cell = self.world.grid[self.grid_x][self.grid_y]
        if not cell.isHighway or cell.highway is None:
            return self.R_INVALID, False
        hw = cell.highway
        if hw.orientation == "horizontal":
            self.hw_row, self.hw_col = 1, 0
        else:
            self.hw_row, self.hw_col = 0, 1
        self.in_highway = True
        if hw.has_car(self.hw_row, self.hw_col):
            return self.R_HIT_CAR, False
        return self.R_HIGHWAY_STEP, False

    def _highway_action(self, action):
        """Apply a highway movement action, attempt to exit or move diagonally, and return the resulting reward and termination status."""
        hw = self.world.grid[self.grid_x][self.grid_y].highway
        if action == 5:
            return self._exit_highway()
        if action == 4:
            return self.R_INVALID, False
        if action == 6:
            return self._diagonal_move(hw)
        dr, dc = self._hw_direction(action, hw)
        nr, nc = self.hw_row + dr, self.hw_col + dc
        if not (0 <= nr < 3 and 0 <= nc < 3):
            return self._exit_highway()
        reward = self.R_HIGHWAY_STEP
        if hw.has_car(nr, nc):
            reward = self.R_HIT_CAR
        self.hw_row, self.hw_col = nr, nc
        return reward, False

    def _diagonal_move(self, hw):
        """Attempt a diagonal move in the highway, checking for bounds and cars, and return the resulting reward and termination status."""
        if hw.orientation == "horizontal":
            lane = self.np_random.choice([-1, 1])
            nr, nc = self.hw_row + lane, self.hw_col + 1
        else:
            lane = self.np_random.choice([-1, 1])
            nr, nc = self.hw_row + 1, self.hw_col + lane
        if not (0 <= nr < 3 and 0 <= nc < 3):
            return self.R_INVALID, False
        reward = self.R_DIAGONAL
        if hw.has_car(nr, nc):
            reward = self.R_HIT_CAR
        self.hw_row, self.hw_col = nr, nc
        return reward, False

    def _exit_highway(self):
        """Exit the highway, checking for cars in the exit cell, and return the resulting reward and termination status."""
        self.in_highway = False
        self.hw_row = self.hw_col = 0
        return 1, False

    @staticmethod 
    def _hw_direction(action, hw):
        """Get the row and column deltas for a highway movement action based on the highway's orientation."""
        if hw.orientation == "horizontal":
            return [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        return [(-1, 0), (1, 0), (0, -1), (0, 1)][action]

    def _in_bounds(self, x, y):
        """Check if the given coordinates are within the bounds of the grid."""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    @property
    def agent_pos(self):
        """Return the current position of the agent in the grid as a tuple (grid_x, grid_y)."""
        return (self.grid_x, self.grid_y)