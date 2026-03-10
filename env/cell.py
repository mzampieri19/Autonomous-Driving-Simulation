class Cell:
    """
    Represents a cell in the grid environment. It can be a regular cell or part of a highway.
    """
    def __init__(self, x, y):
        """Initialize a cell with its coordinates. By default, it's a regular cell without a car or highway."""
        self.x = x
        self.y = y
        self.isHighway = False
        self.connections = []
        self.highway = None  # Will hold Highway object if isHighway is True
        self.car = None  # For non-highway cells