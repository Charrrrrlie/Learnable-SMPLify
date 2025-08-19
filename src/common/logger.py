import logging

_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_DATEFMT = "%m/%d/%Y %H:%M:%S"


def fileHandler(path, format, datefmt, mode="w"):
    handler = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler

  
def getLogger(
    name=None,
    path=None,
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
):
    logging.basicConfig(filename=path, 
                    level=level, 
                    format=format,
                    datefmt=datefmt)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger