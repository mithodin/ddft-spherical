import numpy as np

from typing import Tuple, List, Dict


class Logger:
    """
    Base class to implement logging for the simulation.
    Extend this class to implement your own logger.
    """
    def log_state(self, body: List[Tuple[str, np.ndarray]], header: Dict[str, float] = None) -> None:
        pass
