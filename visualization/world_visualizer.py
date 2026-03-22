import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from env.car import CAR_COLORS
from env.world import World

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

class WorldVisualizer:
    """
    Interactive matplotlib visualisation for the World grid using sprite images.

    Assets expected in assets/:
        {color}_car_horizontal.png / {color}_car_vertical.png
        horizontal_highway.png / vertical_highway.png
        non_highway_cell.png
    """

    FALLBACK = {
        "start":         "#4CAF50",
        "goal":          "#F44336",
        "highway_h":     "#2196F3",
        "highway_v":     "#9C27B0",
        "empty":         "#E0E0E0",
        "agent":         "#FFEB3B",
        "highway_car":   "#D32F2F",
        "highway_empty": "#BBDEFB",
    }

    def __init__(self, world: World, agent_pos=None):
        """Initialize the visualizer with a World instance and an agent position."""
        self.world      = world
        self.agent_pos  = agent_pos
        self.selected   = None
        self._images    = self._load_images()

        plt.ion()
        self.fig, (self.ax_grid, self.ax_detail) = plt.subplots(
            1, 2, figsize=(14, 8),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.draw_grid()
        self._update_stats()

    def _load_images(self):
        """Load all required images into a dictionary, using None for missing assets."""
        files = {
            "highway_h": "horizontal_highway.png",
            "highway_v": "vertical_highway.png",
            "empty":     "non_highway_cell.png",
        }
        for color in CAR_COLORS:
            files[f"car_{color}_h"] = f"{color}_car_horizontal.png"
            files[f"car_{color}_v"] = f"{color}_car_vertical.png"

        images = {}
        for key, fname in files.items():
            path = os.path.join(ASSETS_DIR, fname)
            if os.path.exists(path):
                images[key] = mpimg.imread(path)
            else:
                print(f"[WorldVisualizer] Missing asset: {path}")
                images[key] = None
        return images

    def draw_grid(self):
        """Draw the world grid with highways, cars, and agent position using loaded images or fallback colors."""
        self.ax_grid.clear()
        gs = self.world.grid_size
        for x in range(gs):
            for y in range(gs):
                self._draw_cell(x, y, gs)
        self.ax_grid.set_xlim(0, gs)
        self.ax_grid.set_ylim(0, gs)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_title("World Grid  (click a cell for details)")
        self.ax_grid.set_xlabel("X")
        self.ax_grid.set_ylabel("Y")

    def _draw_cell(self, x, y, gs):
        """Draw a single cell at (x, y) using an image if available, otherwise a colored rectangle. Agent and special cells have priority."""
        display_y = gs - 1 - y
        image_key, fallback_key, label = self._resolve_cell(x, y)
        img = self._images.get(image_key) if image_key else None

        if img is not None:
            self.ax_grid.imshow(
                img,
                extent=[x, x + 1, display_y, display_y + 1],
                aspect="auto", zorder=1,
            )
            rect = patches.Rectangle(
                (x, display_y), 1, 1,
                linewidth=0.5, edgecolor="black", facecolor="none", zorder=2,
            )
        else:
            color = self.FALLBACK.get(fallback_key, "#E0E0E0")
            rect = patches.Rectangle(
                (x, display_y), 1, 1,
                linewidth=1, edgecolor="black", facecolor=color, zorder=1,
            )

        self.ax_grid.add_patch(rect)
        if label:
            self.ax_grid.text(
                x + 0.5, display_y + 0.5, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=3,
            )

    def _resolve_cell(self, x, y):
        """Determine the appropriate image key, fallback color key, and label for a cell at (x, y) based on its contents and agent position."""
        cell = self.world.grid[x][y]
        if self.agent_pos and (x, y) == self.agent_pos:
            return None, "agent", "A"
        if (x, y) == self.world.start:
            return None, "start", "S"
        if (x, y) == self.world.goal:
            return None, "goal", "G"
        if cell.isHighway:
            key = "highway_v" if (cell.highway and
                                   cell.highway.orientation == "vertical") else "highway_h"
            return key, key, "H"
        if cell.car is not None:
            return f"car_{cell.car.color}_h", "empty", "C"
        return "empty", "empty", ""

    def _on_click(self, event):
        """Handle click events on the grid to show cell details in the side panel."""
        if event.inaxes != self.ax_grid:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = self.world.grid_size - 1 - int(event.ydata)
        if 0 <= x < self.world.grid_size and 0 <= y < self.world.grid_size:
            self.selected = (x, y)
            self._show_cell_info(x, y)
            self.fig.canvas.draw_idle()

    def _show_cell_info(self, x, y):
        """Display detailed information about the cell at (x, y) in the side panel, including highway details if applicable."""
        self.ax_detail.clear()
        cell  = self.world.grid[x][y]
        lines = [f"Position: ({x}, {y})", f"Highway: {cell.isHighway}"]

        if (x, y) == self.world.start:
            lines.append("Type: START")
        elif (x, y) == self.world.goal:
            lines.append("Type: GOAL")

        if cell.isHighway and cell.highway:
            lines += [
                f"Orientation: {cell.highway.orientation}",
                f"Connections: {len(cell.connections)}",
                f"Cars in highway: {sum(sum(r) for r in cell.highway.grid)}",
            ]
            self._draw_highway_detail(cell.highway)
        elif cell.isHighway:
            lines.append("(terminal cell — no obstacle grid)")
            self.ax_detail.text(0.5, 0.5, "Terminal highway cell",
                                ha="center", va="center", fontsize=12)
            self.ax_detail.set_xlim(0, 1); self.ax_detail.set_ylim(0, 1)
        else:
            lines.append(f"Has car: {'Yes (' + cell.car.color + ')' if cell.car else 'No'}")
            self.ax_detail.text(0.5, 0.5, "Not a highway cell",
                                ha="center", va="center", fontsize=12)
            self.ax_detail.set_xlim(0, 1); self.ax_detail.set_ylim(0, 1)

        self.ax_detail.set_title("\n".join(lines), fontsize=10, loc="left")

    def _draw_highway_detail(self, highway):
        """Draw a 3x3 grid representing the highway's internal structure, showing car placements."""
        for row in range(3):
            for col in range(3):
                has_car = highway.grid[row][col]
                color   = self.FALLBACK["highway_car" if has_car else "highway_empty"]
                self.ax_detail.add_patch(patches.Rectangle(
                    (col, 2 - row), 1, 1,
                    linewidth=2, edgecolor="black", facecolor=color,
                ))
                if has_car:
                    self.ax_detail.text(col + 0.5, 2 - row + 0.5, "C",
                                        ha="center", va="center",
                                        fontsize=14, fontweight="bold", color="white")
        self.ax_detail.set_xlim(0, 3)
        self.ax_detail.set_ylim(0, 3)
        self.ax_detail.set_aspect("equal")

    def _update_stats(self):
        """Update the figure title with current world statistics such as grid size, number of highways, and cars."""
        s = self.world.get_highway_stats()
        self.fig.suptitle(
            f"Grid: {self.world.grid_size}×{self.world.grid_size}  |  "
            f"Highways: {s['total_highways']} "
            f"(H:{s['horizontal_highways']}, V:{s['vertical_highways']})  |  "
            f"Highway cars: {s['total_highway_cars']}",
            fontsize=11,
        )

    def update(self, agent_pos=None):
        """Update the visualization with a new agent position and refresh the grid and stats."""
        self.agent_pos = agent_pos
        self.draw_grid()
        self._update_stats()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self):
        """Display the visualization window and block until it is closed."""
        plt.tight_layout()
        plt.show(block=True)

    def show_non_blocking(self):
        """Display the visualization window without blocking, allowing for dynamic updates."""
        plt.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, filename="world_visualization.png"):
        """Save the current visualization to a file with the given filename."""
        self.fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved → {filename}")

    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)