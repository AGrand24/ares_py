import numpy as np
import pandas as pd


def coords_load(fp):

    df = pd.read_csv(fp, header=None).iloc[:, :4]
    coords = df.values.astype(float)
    coords = np.round(coords, 2)
    return coords


def coords_ld2d(coords):

    diff = np.diff(coords[:, [1, 2]], axis=0, prepend=np.nan)
    ld_hor = np.power(diff, 2)
    ld_hor = np.sum(ld_hor, axis=1)
    ld_hor = ld_hor**0.5
    ld_hor = np.nancumsum(ld_hor)
    coords = np.column_stack([coords, ld_hor])
    return coords


def coords_interpolate(coords):

    l1 = np.arange(len(coords[:, 0]))
    l2 = np.arange(0, len(coords[:, 0]) - 1 + 0.1, 0.1)
    # index = np.arange(0, np.max(coords[:, 0]), step=0.5)
    interpolated = []
    for i in range(0, coords.shape[1]):
        interpolated.append(np.interp(l2, l1, coords[:, i]))

    coords_int = np.column_stack(interpolated)
    coords_int[:, 0] = np.round(coords_int[:, 0], 1)
    return coords_int


def coord_merge(data, coords_int):

    df_l = data.copy()[["ld", "doi"]]
    df_r = pd.DataFrame(coords_int[:, 1:], index=coords_int[:, 0])

    df = pd.merge(df_l, df_r, "left", left_on="ld", right_index=True)
    df.columns = ["ld", "z", "x", "y", "z0", "ld_hor"]

    df["z"] = df["z"] + df["z0"]

    df.iloc[:, 1:] = np.round(df.iloc[:, 1:], 2)

    data[df.columns] = df[df.columns]

    return data
