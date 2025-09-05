import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import Polygon, LineString
from plotly.colors import sample_colorscale


def get_xy_ranges(data, sec, scale, graph_height):

    graph_height *= scale

    xrange = []
    yrange = []
    geom = []

    for line in data["ID_line"].unique():
        df1 = sec.loc[sec["ID_line"] == line].copy()
        df2 = data.loc[data["ID_line"] == line].copy()

        xoffset = int(line) * 10000
        df1["ld_hor"] += xoffset
        xrange.append((df1["ld_hor"].min(), df1["ld_hor"].max()))

        ymax = 5 * ((df2["z0"].max() // 5) + 1) + 20
        ymin = ymax - graph_height
        yrange.append((ymin, ymax))

    return xrange, yrange


def get_ylog(y, ymax, scale, grph_size):
    y = np.clip(y, 1, ymax)
    y = np.log10(y)
    ymax = np.log10(ymax)
    y = y / ymax
    y *= grph_size * (scale / 1000)
    return y


def get_extents(lines, xrange, yrange, scale):

    geom = []
    width = []
    ymin = []
    ymax = []
    range = []
    for x, y in zip(xrange, yrange):

        geom.append(Polygon(((x[0], y[0]), (x[0], y[1]), (x[1], y[1]), (x[1], y[0]))))
        width.append(round((x[1] - x[0]) / scale) + 35)
        ymin.append(y[0])
        ymax.append(y[1])
        range.append(y[1] - y[0])
    gdf = gpd.GeoDataFrame(data={"ID_line": lines}, geometry=geom, crs=8353)
    gdf["width"] = width
    gdf["ymin"] = ymin
    gdf["ymax"] = ymax
    gdf["yrange"] = range
    return gdf


def get_levels(df):
    df["lab"] = "z= " + np.abs(np.round(df["doi"])).astype(int).astype(str).str.zfill(2)
    df["lab"] += ", a= " + df["a"].astype(str)
    df["lab"] += ", n= " + df["n"].astype(str)

    cols = ["ld_hor", "res", "lab"]
    levels = df.groupby("doi")[cols].apply(np.array).to_list()
    levels = levels[::3]
    return levels


def get_qc_grph_single(levels, xoffset, yrange, graph_height, ymax):
    colors = sample_colorscale("Turbo_r", np.linspace(0, 1, len(levels)))
    levels
    geom = []
    clr = []
    name = []
    for i, lvl in enumerate(levels):
        x = lvl[:, 0].astype(float) + xoffset
        y = lvl[:, 1].astype(float)
        y = get_ylog(y, ymax, 500, graph_height)
        y = y + yrange
        geom.append(LineString(zip(x, y)))

        name.append(lvl[0, 2])
        clr.append(colors[i])

    gdf = gpd.GeoDataFrame(data={"lvl": name, "clr": clr}, geometry=geom, crs=8353)
    return gdf


def get_qc_grph_ls(prj):

    gdf = []
    for line, ert in prj.ert.items():
        df = ert.data.copy()
        levels = get_levels(df)
        xoffset = prj.layout_qc["xrange"].loc[line, "min"]
        yrange = prj.layout_qc["yrange"].loc[line, "min"]

        gdf_tmp = get_qc_grph_single(
            levels,
            xoffset,
            yrange,
            prj.layout_qc["graph_height"],
            prj.layout_qc["res_max"],
        )
        gdf_tmp["ID_line"] = line
        gdf.append(gdf_tmp)

    gdf = pd.concat(gdf).reset_index(drop=True)
    gdf.to_file(prj.fp["ert"], layer="ert2d_qc_graphs_ls", engine="pyogrio")
    return gdf
