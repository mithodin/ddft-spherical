import numpy as np

from typing import Tuple, List, Dict, Any


class Logger:
    """
    Base class to implement logging for the simulation.
    Extend this class to implement your own logger.
    Always use in a 'with' statement
    """
    def log_state(self: 'Logger', body: List[Tuple[str, np.ndarray]], header: Dict[str, float] = None) -> None:
        pass

    """
    This function is called once at the beginning of the simulation.
    Can be used e.g. to open a file handle
    """
    def __enter__(self: 'Logger') -> 'Logger':
        return self

    """
    This function is called once after the end of the simulation.
    Here, you could e.g. close a file handle
    """
    def __exit__(self: 'Logger', exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
