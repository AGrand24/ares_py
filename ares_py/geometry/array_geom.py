import numpy as np
import pandas as pd


def calc_geom_factor(electrodes: np.ndarray, round=False):

    c1 = electrodes[:, 0]
    c2 = electrodes[:, 1]
    p1 = electrodes[:, 2]
    p2 = electrodes[:, 3]

    r1 = np.abs(c1 - p1)
    r2 = np.abs(c2 - p1)
    r3 = np.abs(c1 - p2)
    r4 = np.abs(c2 - p2)

    k = 2 * np.pi / (1 / r1 - 1 / r2 - 1 / r3 + 1 / r4)

    if round == True:
        k = np.round(k).astype(np.int64)

    return k


def get_2d_geom_factor(df):
    geom_factor = df["geom_factor"].copy()
    geom_factor_2d = pd.Series(geom_factor.copy().explode())

    geom_factor_2d = geom_factor_2d.replace("", np.nan)
    geom_factor_2d = geom_factor_2d.astype(float)

    rows = len(geom_factor)
    cols = len(geom_factor.iloc[0])
    geom_factor_2d = np.array(geom_factor_2d).reshape(rows, cols)
    return geom_factor_2d
