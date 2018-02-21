#!/usr/bin/env python3
"""National Bureau of Statistics of China: Urban Life & Price Yearbooks.

Under development.
"""
from os.path import dirname, join

import pandas as pd

DATA_PATH = join(dirname(__file__), 'cn_nbs', 'ulpy')


def load():
    return pd.read_csv(join(DATA_PATH, '2008_1-4-6.csv'))
