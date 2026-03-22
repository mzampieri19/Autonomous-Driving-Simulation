import numpy as np
from .cell import Cell
from .highway import Highway
from .car import Car


class World:
    """
    Represents the grid environment with highways and cars. The world is initialized with a specified grid size, number of highway branches, maximum branch length, and number of cars.
    """
    def __init__(self, grid_size, num_branches, max_length, num_cars=5, highway_max_cars=4):
        """Initialize the world with highways and cars."""
        self.grid_size = grid_size
        self.grid = [[Cell(x, y) for y in range(grid_size)] for x in range(grid_size)] # Create grid of Cell objects
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.highway_max_cars = highway_max_cars
        self.create_highways(num_branches=num_branches, max_length=max_length)
        self._add_cars(num_cars=num_cars)

    def create_highways(self, num_branches=5, max_length=4):
        """Create the main highway path and add branches, then build connections and configure highway grids."""
        self._create_main_path()
        self._add_branches(num_branches=num_branches, max_length=max_length)
        self._build_connections()
        self._create_highway_grids()
        self._configure_terminal_cells()

    def _configure_terminal_cells(self):
        """Ensure start and goal cells are highways and have no cars."""
        for x, y in (self.start, self.goal):
            self.grid[x][y].highway = None

    def _create_main_path(self):
        """Create the main highway path from start to goal by randomly going down or right."""
        x, y = self.start
        self.grid[x][y].isHighway = True
        while (x, y) != self.goal:
            can_right = x < self.goal[0]
            can_up    = y < self.goal[1]
            if can_right and can_up:
                nx, ny = (x + 1, y) if np.random.random() < 0.5 else (x, y + 1)
            elif can_right:
                nx, ny = x + 1, y
            else:
                nx, ny = x, y + 1
            self.grid[nx][ny].isHighway = True
            x, y = nx, ny

    def _add_branches(self, num_branches, max_length):
        """Add branches to the main highway path while ensuring they do not create fully blocked rows or columns."""
        highway_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if self.grid[x][y].isHighway
        ]
        for _ in range(num_branches):
            if not highway_cells:
                break
            x, y = highway_cells[np.random.randint(len(highway_cells))]
            for _ in range(max_length):
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                np.random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                            and not self.grid[nx][ny].isHighway):
                        self.grid[nx][ny].isHighway = True
                        highway_cells.append((nx, ny))
                        x, y = nx, ny
                        break
                else:
                    break

    def _build_connections(self):
        """Build connections between adjacent highway cells for pathfinding and orientation determination."""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid[x][y]
                cell.connections = []
                if cell.isHighway:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                                and self.grid[nx][ny].isHighway):
                            cell.connections.append((nx, ny))

    def _create_highway_grids(self):
        """Create Highway objects for each highway cell based on their connections to determine orientation and car placement."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid[x][y]
                if cell.isHighway:
                    orientation = self._determine_orientation(x, y, cell.connections)
                    cell.highway = Highway(
                        orientation=orientation,
                        max_cars=self.highway_max_cars,
                    )

    def _determine_orientation(self, x, y, connections):
        """Determine the orientation of a highway cell based on its connections."""
        horizontal = sum(1 for nx, ny in connections if ny == y)
        vertical   = sum(1 for nx, ny in connections if nx == x)
        return "vertical" if vertical > horizontal else "horizontal"

    def _add_cars(self, num_cars):
        """Randomly place cars in non-highway cells while ensuring start and goal cells are not occupied."""
        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if not self.grid[x][y].isHighway
            and (x, y) not in (self.start, self.goal)
        ]
        np.random.shuffle(candidates)
        for x, y in candidates[:num_cars]:
            self.grid[x][y].car = Car(x, y, self)

    def display(self, show_orientation=False):
        """Print the grid representation of the world, showing highways, cars, and highway orientations."""
        for y in range(self.grid_size - 1, -1, -1):
            row = ""
            for x in range(self.grid_size):
                cell = self.grid[x][y]
                if (x, y) == self.start:
                    row += "S "
                elif (x, y) == self.goal:
                    row += "G "
                elif cell.isHighway:
                    if show_orientation and cell.highway:
                        row += "- " if cell.highway.orientation == "horizontal" else "| "
                    else:
                        row += "H "
                elif cell.car is not None:
                    row += "C "
                else:
                    row += ". "
            print(row)
        if show_orientation:
            print("\nLegend: S=Start G=Goal -=H.Highway |=V.Highway C=Car .=Empty")

    def get_highway_stats(self):
        """Calculate and return statistics about the highways and cars in the world."""
        h = v = cars = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.grid[x][y]
                if cell.isHighway and cell.highway:
                    if cell.highway.orientation == "horizontal":
                        h += 1
                    else:
                        v += 1
                    cars += sum(sum(row) for row in cell.highway.grid)
        return {
            "horizontal_highways": h,
            "vertical_highways":   v,
            "total_highways":      h + v,
            "total_highway_cars":  cars,
        }