#!/usr/bin/env python
"""
--------------------------------
project: code
created: 13/07/2018 16:48
---------------------------------

"""
import logging
import pickle


class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def configure_stream_logger(logger, level=logging.DEBUG):
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return None


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_pickle(data, path):
    with open(path, 'wb') as f:
        return pickle.dump(data, f)
