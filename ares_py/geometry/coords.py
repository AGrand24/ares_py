import numpy as np
import pandas as pd
import os


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


def coords_ld3d(coords):

    diff = np.diff(coords[:, [1, 2, 3]], axis=0, prepend=np.nan)
    ld_hor = np.power(diff, 2)
    ld_hor = np.sum(ld_hor, axis=1)
    ld_hor = ld_hor**0.5
    ld_hor = np.nancumsum(ld_hor)
    coords = np.column_stack([coords, ld_hor])
    return coords


def coords_interpolate(coords):
    l1 = coords[:, 0]
    step = 0.1
    l2 = np.arange(0, coords[:, 0].max() + step, step)

    interpolated = []
    for i in range(0, coords.shape[1]):
        interpolated.append(np.interp(l2, l1, coords[:, i]))

    coords_int = np.column_stack(interpolated)
    coords_int[:, 0] = np.round(coords_int[:, 0], 1)
    return coords_int


def coord_merge(ert):
    data = ert.data.copy()
    coords_int = ert.coords_int.copy()

    df_l = data.copy()[["ld", "doi"]]
    df_r = pd.DataFrame(coords_int[:, 1:], index=coords_int[:, 0])

    df = pd.merge(df_l, df_r, "left", left_on="ld", right_index=True)
    df.columns = ["ld", "topo", "x", "y", "z0_topo", "ld_hor"]

    df["topo"] = df["topo"] + df["z0_topo"]

    cols = df.columns[1:]
    data = data.reset_index(drop=True)
    df = df.reset_index(drop=True)
    data[cols] = df[cols]

    return data


def coords_merge_sections(ert):

    df_l = pd.DataFrame(ert.sec[:, :], index=ert.sec[:, 0] * ert.el_space)
    df_r = pd.DataFrame(ert.coords_int[:, 1:], index=ert.coords_int[:, 0])

    df_l

    df = pd.merge(df_l, df_r, "left", left_index=True, right_index=True)

    df = df.reset_index()
    df.columns = ["ld", "n_el", "sec", "n_sec", "x", "y", "z0_topo", "ld_hor"]
    df = df.dropna(subset="x")
    df["dtm"] = 0
    df["dtm_dist"] = 0
    return df


def coords_get_z(df, data_type, zmode="dtm"):

    if zmode == "dtm":
        z = "dtm"
    else:
        z = "topo"
        df["z0_dtm"] = df["z0_topo"].max()
        df["dtm_dist"] = 0

    df["z0"] = df[f"z0_{z}"]

    if data_type == "data":
        df["z"] = df["z0"] + df["doi"]
        df["dtm"] = df["z0_dtm"] + df["doi"]
        cols = ["z", "z0", "topo", "dtm", "z0_dtm", "z0_topo", "dtm_dist"]
    else:
        cols = ["z0", "z0_topo", "z0_dtm", "dtm_dist"]

    df[cols] = df[cols].round(2)

    return df


def coords_create_csv(ert):

    if not os.path.exists(ert.fp["in_topo"]):
        electrodes = ert.data.iloc[:, :4].values
        electrodes = np.unique(electrodes)

        data = [electrodes, electrodes]
        data.append(np.full(shape=(data[0].shape[0], 2), fill_value=0.0))

        data = np.column_stack(data)

        df = pd.DataFrame(data)
        df.to_csv(ert.fp["in_topo"], index=False, header=None)

        print(f"\tTopo .csv not found created new csv: {ert.fp['in_topo']} ")
