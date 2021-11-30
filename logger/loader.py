from enum import unique, Enum
from typing import Dict, Union, Tuple

from util import log
from .logger import Logger
from .ascii_logger import AsciiLogger, StdoutLogger


@unique
class LoggerID(Enum):
    Stdout = "stdout"
    Ascii = "ascii"


def load_logger(logger_config: Union[str, Dict[str, str]]) -> Logger:
    logger_config, logger_id = get_id_and_config(logger_config)
    if logger_id == LoggerID.Ascii:
        return load_ascii_logger(logger_config)
    return StdoutLogger()


def load_ascii_logger(logger_config: Dict[str, str]) -> AsciiLogger:
    log(" > using ascii logger")
    try:
        log_filename = logger_config["output"]
    except KeyError:
        log(" > found no filename for ascii logger, using \"out/diffusion.dat\"")
        log_filename = "out/diffusion.dat"
    return AsciiLogger(log_filename)


def get_id_and_config(logger_config: Union[str, Dict[str, str]]) -> Tuple[Dict[str, str], LoggerID]:
    logger_config_dict: dict = dict() if isinstance(logger_config, str) else logger_config
    try:
        logger_name = logger_config if isinstance(logger_config, str) else logger_config["name"]
    except KeyError:
        log(" > no logger name found, using stdout")
        logger_name = "stdout"
    try:
        logger_id: LoggerID = LoggerID(logger_name)
    except ValueError:
        log(" > configured logger \"{}\" not found, using stdout instead".format(logger_name))
        logger_id = LoggerID.Stdout
    return logger_config_dict, logger_id
