import sys
import numpy as np

from typing import Tuple, List, Dict, TextIO, Any
from logger.logger import Logger


class AsciiLogger(Logger):
    """
    This utility logs the passed state in ascii format to a file
    """
    _file_name: str
    _output: TextIO

    def __init__(self: 'AsciiLogger', file_name: str) -> None:
        self._file_name = file_name

    def __enter__(self: 'AsciiLogger') -> 'AsciiLogger':
        self._output = open(self._file_name, mode='w')
        self._output.__enter__()
        return self

    def __exit__(self: 'AsciiLogger', *kwargs: Any) -> None:
        self._output.__exit__(*kwargs)

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
        np.savetxt(self._output, data, header=column_header, footer="\n", comments="")


class StdoutLogger(AsciiLogger):
    """
    This utility logs the passed state on stdout with the same format as AsciiLogger
    """
    def __init__(self: 'StdoutLogger') -> None:
        super(StdoutLogger, self).__init__('')

    def __enter__(self: 'StdoutLogger') -> 'StdoutLogger':
        self._output = sys.stdout
        return self

    def __exit__(self: 'StdoutLogger', *kwargs: Any) -> None:
        pass
