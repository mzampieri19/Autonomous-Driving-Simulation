import numpy as np

class Highway:
    """
    Represents a highway cell's internal 3x3 grid with cars as obstacles.

    - Horizontal highways: no row may be fully blocked (agent moves left/right)
    - Vertical highways:   no column may be fully blocked (agent moves up/down)
    """

    MAX_CELLS = 9  # 3x3 grid

    def __init__(self, orientation="horizontal", max_cars=4): # Default to 4 cars
        """Initialize the highway with a given orientation and randomly place cars while respecting blocking rules."""
        self.orientation = orientation
        self.grid = [[False] * 3 for _ in range(3)]
        self._place_cars(max_cars)

    def _place_cars(self, max_cars):
        """Randomly place cars in the 3x3 grid while respecting the blocking rules."""
        if max_cars <= 0:
            return
        max_cars = min(max_cars, self.MAX_CELLS) # Ensure no more than 9 cars are placed
        num_cars = np.random.randint(1, max_cars + 1)
        positions = [(r, c) for r in range(3) for c in range(3)]
        np.random.shuffle(positions)
        placed = 0
        for row, col in positions:
            if placed >= num_cars:
                break
            self.grid[row][col] = True
            if self._is_blocked():
                self.grid[row][col] = False
            else:
                placed += 1

    def _row_blocked(self, row):
        """Check if a specific row is fully blocked."""
        return all(self.grid[row])

    def _col_blocked(self, col):
        """Check if a specific column is fully blocked."""
        return all(self.grid[row][col] for row in range(3))

    def _is_blocked(self):
        """Check if the current grid configuration violates the blocking rules."""
        if self.orientation == "horizontal":
            return any(self._row_blocked(r) for r in range(3))
        return any(self._col_blocked(c) for c in range(3))

    def is_passable(self):
        """Check if the highway cell is passable (not fully blocked)."""
        return not self._is_blocked()

    def has_car(self, row, col):
        """Check if there's a car at the specified position."""
        return self.grid[row][col]

    def display(self):
        """Print the highway grid for visualization."""
        print(f"Orientation: {self.orientation}")
        for row in self.grid:
            print(" ".join("C" if c else "." for c in row))

    def __repr__(self):
        """String representation for debugging."""
        lines = [f"Orientation: {self.orientation}"]
        for row in self.grid:
            lines.append("".join("C" if c else "." for c in row))
        return "\n".join(lines)