import random

CAR_COLORS = ["orange", "blue", "green", "pink", "yellow"]

class Car:
    """
    Represents a car in the grid environment. Each car has a position (x, y) and a color.
    """
    def __init__(self, x, y, world):
        """Initialize a car with its coordinates and the world it belongs to. The car is assigned a random color."""
        self.x = x
        self.y = y
        self.world = world
        self.color = random.choice(CAR_COLORS)