import sys


def log(*args):
    print(*args, file=sys.stderr)


def get_functional_config(config: dict) -> (str, str):
    try:
        base_functional = config["functional"]["base"]
    except KeyError:
        base_functional = None
    try:
        variant = config["functional"]["variant"]
    except KeyError:
        variant = None
    return base_functional, variant
