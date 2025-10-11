import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from ares_py.tools.geometry_tools import pt_to_ls


def export_topo_ls(prj):

    df = prj.topo[["ID_line", "ld_hor", "z0", "ld"]].copy()
    df["ld_hor"] += df["ID_line"].astype(float) * 10000

    gdf_ls = pt_to_ls(df, x="ld_hor", y="z0", order="ld", groupby="ID_line", crs=8353)
    gdf_ls.to_file(prj.fps["ert"], layer="ert2d_topo_ls", engine="pyogrio")


def export_topo_pt(prj):

    df = prj.topo[["ID_line", "ld_hor", "z0", "ld"]].copy()
    df["ld_hor"] += df["ID_line"].astype(float) * 10000

    geom = gpd.points_from_xy(x=df["ld_hor"], y=df["z0"])
    gdf_pt = gpd.GeoDataFrame(df, geometry=geom, crs=8353)
    gdf_pt.to_file(prj.fps["ert"], layer="ert2d_topo_pt", engine="pyogrio")


def export_crd_csv(df, line, prj):
    fp = Path(prj.fps["crd"], str(line).zfill(3) + "_plan.csv")
    df.to_csv(fp)


def export_crd_man(prj):

    gdf = gpd.read_file(prj.fps["ert"], layer="crd_man_proc_pt")
    gdf = gdf.set_index("ld")

    for ert in prj.ert.values():
        line = int(ert.line)
        tmp = gdf.copy().loc[gdf["ID_line"] == line]
        tmp = tmp[["ld_hor", "x", "y", "z0"]]
        tmp = tmp.add_suffix("_man")
        tmp = tmp.round(2)
        tmp = tmp.reset_index()

        fp_out = Path(prj.fps["crd"], f"{ert.line}_man.csv")
        tmp.to_csv(fp_out, index=False)


def coords_load(fp):

    df = pd.read_csv(fp, header=None).iloc[:, :4]
    coords = df.values.astype(float)
    coords = np.round(coords, 2)
    return coords


def calc_line_distance(x, y, z=None):
    try:
        if z == None:
            z = np.full(x.shape, 0)
    except:
        pass

    coords = np.column_stack([x, y, z])
    diff = np.diff(coords, axis=0, prepend=0)
    ld = np.power(diff, 2)
    ld = np.sum(ld, axis=1)
    ld = ld**0.5
    ld = np.nancumsum(ld)
    ld = ld - np.nanmin(ld)
    return ld


def calc_topo_ld_hor(ld, z0):
    # calc ld_hor for topo -takes into account only ld and z0 not XYZ,
    # considers line to be straight in XY plane
    crd = np.column_stack([ld, z0])
    diff = np.diff(crd, axis=0)
    diff = np.power(diff, 2)
    ld_hor = (diff[:, 0] - diff[:, 1]) ** 0.5
    ld_hor = [0] + list(ld_hor)
    ld_hor = np.cumsum(ld_hor)
    return ld_hor


def coords_interpolate(reference, values):
    l1 = reference
    step = 0.1
    l2 = np.arange(0, reference.max() + step, step)

    interpolated = []
    values = np.column_stack([reference, values])
    for i in range(0, values.shape[1]):
        interpolated.append(np.interp(l2, l1, values[:, i]))

    interpolated = np.column_stack(interpolated)
    return interpolated


def process_line_distance(gdf):
    gdf_out = []

    for line in gdf["ID_line"].unique():
        tmp = gdf.loc[gdf["ID_line"] == line].copy()
        id_line = tmp["ID_line"].iloc[0]
        el_space = tmp["el_space"].iloc[0]

        crd = tmp.get_coordinates(include_z=True)
        ld = calc_line_distance(x=crd["x"], y=crd["y"], z=crd["z"])

        crd_int = coords_interpolate(reference=ld, values=crd)
        crd_int = np.round(crd_int, 2)

        df = pd.DataFrame(crd_int)
        df.columns = ["ld", "x", "y", "z0"]

        df["ID_line"] = id_line
        df["el_space"] = el_space

        mask = df["ld"] % 1 == 0
        df = df.loc[mask]

        df["ld_hor"] = calc_line_distance(df["x"], df["y"])
        df["ld_hor"] = df["ld_hor"].round(2)
        gdf_out.append(df)

    gdf_out = pd.concat(gdf_out)
    gdf_out = gdf_out.reset_index(drop=True)

    return gdf_out


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


def create_csv_flat(df, fp):
    el_max = df[["c1", "c2", "p1", "p2"]].max().max()

    ld = np.arange(0, el_max, 1)
    data = np.column_stack([ld, ld, ld, np.full((ld.shape[0], 2), 0)])

    df = pd.DataFrame(data)
    df.columns = ["ld", "ld_hor", "x", "y", "z0"]
    df = df.set_index("ld")
    df = df.add_suffix("_flat")
    df.to_csv(fp)
