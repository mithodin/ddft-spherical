import numpy as np


class Logger:
    """
    Base class to implement logging for the simulation.
    Extend this class to implement your own logger.
    """
    def log_state(self, body: list[(str, np.array)], header: dict = None) -> None:
        pass
