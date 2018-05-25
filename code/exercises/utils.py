#!/usr/bin/env python
"""
--------------------------------
project: code
created: 11/04/2018 17:46
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
