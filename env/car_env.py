"""
Gymnasium environment for the self-driving car simulation.

Navigation model:
    - Moving onto a highway cell puts the agent in a PENDING state
    - In pending state: action 4 = enter highway, actions 0-3 = skip and move away
    - Inside the sub-grid the agent uses actions 0-3 to navigate the 3x3 grid
    - Walking off the far edge exits back to the grid
    - Walking off the entry edge costs R_BACKWARD_EXIT but is allowed

Observation (7 integers):
    [grid_x, grid_y, in_highway, hw_row, hw_col, orientation, pending_highway]
    orientation:     0=none, 1=horizontal, 2=vertical
    pending_highway: 0=no, 1=standing on highway cell, not yet entered

Actions (5 discrete):
    0=Up  1=Down  2=Left  3=Right  4=Enter highway (only valid when pending)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .world import World

class CarEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    R_GOAL          =  200
    R_HIGHWAY_STEP  =    5
    R_STEP          =   -3
    R_INVALID       =   -1
    R_HIT_CAR       =  -50
    R_BACKWARD_EXIT =   -5
    R_SKIP_HIGHWAY  =   -4   # Penalty for skipping a clear highway

    N_ACTIONS = 5   # Up, Down, Left, Right, Enter highway

    def __init__(self, grid_size=16, num_branches=8, max_length=5,
                 num_cars=12, highway_max_cars=3, max_steps=400,
                 fixed_world=False):
        super().__init__()
        self.grid_size        = grid_size
        self.num_branches     = num_branches
        self.max_length       = max_length
        self.num_cars         = num_cars
        self.highway_max_cars = highway_max_cars
        self.max_steps        = max_steps
        self.fixed_world      = fixed_world

        self.world           = None
        self.grid_x          = 0
        self.grid_y          = 0
        self.in_highway      = False
        self.hw_row          = 0
        self.hw_col          = 0
        self.hw_entry        = None
        self.pending_highway = False
        self._last_action    = 0
        self._steps          = 0
        self._visited        = {}

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.MultiDiscrete(
            [grid_size, grid_size, 2, 3, 3, 3, 2]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.fixed_world or self.world is None:
            self.world = World(
                self.grid_size, self.num_branches, self.max_length,
                num_cars=self.num_cars, highway_max_cars=self.highway_max_cars,
            )
        self.grid_x, self.grid_y = self.world.start
        self.in_highway      = False
        self.hw_row          = 0
        self.hw_col          = 0
        self.hw_entry        = None
        self.pending_highway = False
        self._last_action    = 0
        self._steps          = 0
        self._visited        = {}
        return self._obs(), {}

    def step(self, action):
        self._steps += 1
        reward, terminated = self._apply_action(int(action))

        # Loop penalty hard-capped at -10 after 3 visits to the same state
        s = tuple(self._obs())
        self._visited[s] = self._visited.get(s, 0) + 1
        if self._visited[s] > 3:
            reward += max(-10, -2 * (self._visited[s] - 3))

        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        orientation = 0
        cell = self.world.grid[self.grid_x][self.grid_y]
        if (self.in_highway or self.pending_highway) and cell.highway:
            orientation = 1 if cell.highway.orientation == "horizontal" else 2
        return np.array([
            self.grid_x,
            self.grid_y,
            int(self.in_highway),
            self.hw_row if self.in_highway else 0,
            self.hw_col if self.in_highway else 0,
            orientation,
            int(self.pending_highway),
        ], dtype=np.int64)

    def _apply_action(self, action):
        if self.in_highway:
            return self._highway_action(action)
        if self.pending_highway:
            return self._pending_action(action)
        return self._grid_action(action)

    def _grid_action(self, action):
        """Move on the main grid."""
        if action == 4:
            return self.R_INVALID, False   # Enter only valid when pending

        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
        nx, ny = self.grid_x + dx, self.grid_y + dy

        if not self._in_bounds(nx, ny):
            return self.R_INVALID, False

        self.grid_x, self.grid_y = nx, ny

        if (nx, ny) == self.world.goal:
            return self.R_GOAL, True

        cell = self.world.grid[nx][ny]
        if cell.isHighway and cell.highway is not None:
            self.pending_highway = True
            self._last_action    = action
            return self.R_STEP, False

        return self.R_STEP, False

    def _pending_action(self, action):
        """
        Agent is standing on a highway cell and must decide:
            action 4:   enter the highway sub-grid
            action 0-3: skip — move off in that direction
        """
        if action == 4:
            self.pending_highway = False
            hw = self.world.grid[self.grid_x][self.grid_y].highway
            return self._auto_enter_highway(hw, self._last_action)

        # Skip, move off the highway cell
        self.pending_highway = False
        prev_x, prev_y = self.grid_x, self.grid_y

        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][action]
        nx, ny = self.grid_x + dx, self.grid_y + dy

        if not self._in_bounds(nx, ny):
            return self.R_INVALID, False

        self.grid_x, self.grid_y = nx, ny

        if (nx, ny) == self.world.goal:
            return self.R_GOAL, True

        # Penalize skipping only if entry cell was clear
        hw = self.world.grid[prev_x][prev_y].highway
        entry_map = {0: (2,1), 1: (0,1), 2: (1,2), 3: (1,0)}
        er, ec = entry_map.get(self._last_action, (1, 1))
        if hw and not hw.has_car(er, ec):
            return self.R_SKIP_HIGHWAY, False
        return self.R_STEP, False

    def _auto_enter_highway(self, hw, action):
        """
        Enter from the edge the agent approached from, middle lane.
            0=Up    → entering from bottom (row=2)
            1=Down  → entering from top    (row=0)
            2=Left  → entering from right  (col=2)
            3=Right → entering from left   (col=0)
        """
        if action == 0:
            self.hw_row, self.hw_col = 2, 1
            self.hw_entry = "bottom"
        elif action == 1:
            self.hw_row, self.hw_col = 0, 1
            self.hw_entry = "top"
        elif action == 2:
            self.hw_row, self.hw_col = 1, 2
            self.hw_entry = "right"
        else:
            self.hw_row, self.hw_col = 1, 0
            self.hw_entry = "left"

        self.in_highway = True

        if hw.has_car(self.hw_row, self.hw_col):
            return self.R_HIT_CAR, False
        return self.R_HIGHWAY_STEP, False

    def _highway_action(self, action):
        """
        Navigate inside the 3x3 sub-grid using the same 4 directional actions.
        Walking off the far edge exits back to the grid.
        Walking off the entry edge costs R_BACKWARD_EXIT.
        """
        if action == 4:
            return self.R_INVALID, False   # Enter not valid inside highway

        hw = self.world.grid[self.grid_x][self.grid_y].highway

        dr, dc = self._hw_direction(action, hw)
        nr, nc = self.hw_row + dr, self.hw_col + dc

        if not (0 <= nr < 3 and 0 <= nc < 3):
            return self._auto_exit_highway(
                backwards=self._is_entry_exit(nr, nc)
            )

        reward = self.R_HIGHWAY_STEP
        if hw.has_car(nr, nc):
            reward = self.R_HIT_CAR

        self.hw_row, self.hw_col = nr, nc
        return reward, False

    def _is_entry_exit(self, nr, nc):
        """True if the agent would exit back through the entry edge."""
        if self.hw_entry == "bottom" and nr > 2: return True
        if self.hw_entry == "top"    and nr < 0: return True
        if self.hw_entry == "right"  and nc > 2: return True
        if self.hw_entry == "left"   and nc < 0: return True
        return False

    def _auto_exit_highway(self, backwards=False):
        """Exit the highway back onto the main grid."""
        self.in_highway      = False
        self.pending_highway = False
        self.hw_row          = 0
        self.hw_col          = 0
        self.hw_entry        = None
        return self.R_BACKWARD_EXIT if backwards else self.R_STEP, False

    @staticmethod
    def _hw_direction(action, hw):
        return [(-1, 0), (1, 0), (0, -1), (0, 1)][action]

    def _in_bounds(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    @property
    def agent_pos(self):
        return (self.grid_x, self.grid_y)