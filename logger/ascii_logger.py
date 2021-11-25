import sys
import numpy as np

from typing import Tuple, List, Dict
from logger.logger import Logger


class AsciiLogger(Logger):
    def log_state(self: 'AsciiLogger', body: List[Tuple[str, np.ndarray]], header: Dict[str, float] = None) -> None:
        if header is None:
            header = dict()
        data = np.hstack(tuple(column[1].reshape(-1, 1) for column in body))
        column_header = "# {}".format(" ".join([column[0] for column in body]))
        extra_header = "\n".join(
            ["# {name} = {value:.30f}".format(name=key, value=header[key]) for key in header.keys()]
        )
        if len(extra_header) > 0:
            column_header += "\n" + extra_header
        np.savetxt(sys.stdout.buffer, data, header=column_header, footer="\n", comments="")
