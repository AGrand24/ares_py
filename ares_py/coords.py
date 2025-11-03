import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from ares_py.tools.geometry_tools import pt_to_ls
from ares_py.electrodes import ld2sec
import plotly.graph_objects as go


def parse_ape(fp):
    "Parse data from .ape file for prj.Merge_coordinates()"
    df = pd.read_csv(fp, sep="\t")

    df_int = df.copy()
    df_int = df_int.set_index("ld")
    cols = ["ld"] + df_int.columns.to_list()
    df_int = coords_interpolate(df_int.index.values, df_int, step=1)
    df_int = np.round(df_int, 2)
    df_int = pd.DataFrame(df_int, columns=cols)
    df_int = df_int.loc[df_int["ld"] % 1 == 0]
    return df, df_int


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


def coords_interpolate(reference, values, step=0.1):
    l1 = reference
    step = step
    l2 = np.arange(0, reference.max() + step, step)

    interpolated = []
    values = np.column_stack([reference, values])
    for i in range(0, values.shape[1]):
        interpolated.append(np.interp(l2, l1, values[:, i]))

    interpolated = np.column_stack(interpolated)
    return interpolated


def process_line_distance(gdf):
    gdf_out = []
    crs = gdf.crs

    for line in gdf["ID_line"].unique():
        tmp = gdf.loc[gdf["ID_line"] == line].copy()
        id_line = tmp["ID_line"].iloc[0]
        el_space = tmp["el_space"].iloc[0]

        tmp[["x", "y", "z"]] = tmp.get_coordinates(include_z=True)
        tmp = tmp[["x", "y", "z", "angle"]]

        ld = calc_line_distance(x=tmp["x"], y=tmp["y"], z=tmp["z"])

        crd_int = coords_interpolate(reference=ld, values=tmp)
        crd_int = np.round(crd_int, 2)

        df = pd.DataFrame(crd_int)
        df.columns = ["ld", "x", "y", "z0", "angle"]

        df["ID_line"] = id_line
        df["el_space"] = el_space

        mask = df["ld"] % 1 == 0
        df = df.loc[mask]

        df["ld_hor"] = calc_line_distance(df["x"], df["y"])
        df["ld_hor"] = df["ld_hor"].round(2)
        gdf_out.append(df)

    gdf_out = pd.concat(gdf_out)
    gdf_out = gdf_out.reset_index(drop=True)
    geom = gpd.points_from_xy(gdf_out["x"], gdf_out["y"], gdf_out["z0"])
    gdf_out = gpd.GeoDataFrame(gdf_out, geometry=geom, crs=crs)

    return gdf_out


def gpkg_coord_pt(mode, prj):

    fp = prj.fps["spatial"]

    if mode == "plan":
        layer = "ert_plan_pt"
    elif mode == "crd":
        layer = "ert_pt"
    else:
        fp = "tmp/proc.gpkg"
        layer = "tmp"

    gdf = gpd.read_file("C://tmp/crd_proc_tmp.gpkg")

    gdf_out = process_line_distance(gdf)
    gdf_out = import_el_space(gdf_out.drop(columns="el_space"), prj=prj)

    electrodes = ld2sec(gdf_out["ld"], el_space=gdf_out["el_space"].iloc[0])
    electrodes = pd.DataFrame(electrodes, columns=["el", "sec", "el_sec"])
    gdf_out = pd.concat([gdf_out, electrodes], axis=1)

    gdf_out.to_file(fp, layer=layer)
    print(f"Exported - {fp} - {layer}")
    print(
        gdf_out.groupby("ID_line").agg(
            el_space=("el_space", "min"),
            ld_hor=("ld_hor", "max"),
            ld=("ld", "max"),
        )
    )
    return gdf_out


def gpkg_rec_ls(prj, overwrite=False):
    gdf = gpd.read_file(prj.fps["rec"], layer="rec_ert_pt")

    gdf[["x", "y", "z"]] = gdf.get_coordinates(include_z=True)

    gdf_rec = pt_to_ls(gdf, x="x", y="y", z="z", order="ld", groupby="ID_line")

    layers = gpd.list_layers(prj.fps["spatial"]).values

    gdf_rec = import_el_space(gdf_rec, prj=prj)

    layer_out = "ert_rec_man_ls"
    if overwrite == True or layer_out not in layers:
        gdf_rec.to_file(prj.fps["spatial"], layer=layer_out)
        print(f"Exported - {prj.fps['spatial']} - {layer_out}")
        print("\n")
    return gdf_rec


def import_el_space(gdf, prj):
    "imports el space based on ID_line from spatial gpkg - ert_plan_ls"
    try:
        gdf_plan = gpd.read_file(prj.fps["spatial"], layer="ert_plan_ls")
        gdf_plan = gdf_plan.set_index("ID_line")["el_space"]

        gdf = gdf.set_index("ID_line")
        gdf = gdf.join(gdf_plan).reset_index()
    except:
        print("Plan layer not found - ignoring el spaces in ert_rec_man_ls!")

    return gdf


def plot_coords_3d(gdf):

    fig = go.Figure()

    marker = dict(size=2, color=gdf["ID_line"], colorscale="Spectral")

    custom_data = gdf[["ID_line", "ld", "ld_hor"]].values
    plt = go.Scatter3d(
        x=gdf["x"],
        y=gdf["y"],
        z=gdf["z0"],
        mode="markers",
        marker=marker,
        customdata=custom_data,
        hovertemplate="ID_line:%{customdata[0]}<br>ld:%{customdata[1]}<br>ld_hor:%{customdata[2]}<br>",
    )

    camera = dict(eye=dict(x=3.5, y=3.5, z=3.5))

    fig = fig.update_layout(
        scene=dict(
            aspectmode="data",
            aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=camera,
        margin=dict(t=0, b=0, l=0, r=0),
        width=600,
        height=400,
    )

    fig.add_trace(plt)
    fig.show()
    return fig


def csv_topo(prj):
    gdf = gpd.read_file(prj.fps["spatial"], layer="ert_pt")

    for line in gdf["ID_line"].unique():
        tmp = gdf.loc[gdf["ID_line"] == line]
        tmp = tmp.loc[tmp["ld"] % tmp["el_space"] == 0]

        fp_out = Path(prj.fps["crd"], f"{str(line).zfill(3)}_topo.csv")

        tmp = tmp[["ld", "ld_hor", "z0", "x", "y", "angle"]]
        tmp = tmp.dropna()

        tmp.to_csv(fp_out, index=False)
