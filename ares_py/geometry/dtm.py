import rasterio
import numpy as np
import shapely
import pandas as pd
import geopandas as gpd
from shapely import MultiPoint, LineString
from shapely.ops import nearest_points
from ares_py.coords import coords_interpolate


def tif_read(fp, crs=5514):
    with rasterio.open(fp) as tif:
        z = tif.read(1)
        z = np.flip(z, axis=0)

        coords = tif.bounds
        xlim = (coords[0], coords[2])
        ylim = (coords[1], coords[3])

        x = np.linspace(xlim[0], xlim[1], tif.width)
        y = np.linspace(ylim[0], ylim[1], tif.height)
        x, y = np.meshgrid(x, y)

    return [x, y, z, crs]


def dtm_get_buffer(x, y):
    geom = gpd.points_from_xy(x, y)

    geom = LineString(geom)
    buffer = shapely.buffer(geom, 5)
    return buffer


def tif_clip_line_lim(x, y, tif):
    df = pd.DataFrame(np.column_stack([d.flatten() for d in tif]))

    xlim = [x.min(), x.max()]
    ylim = [y.min(), y.max()]

    mask2 = df[0] <= xlim[1]
    mask1 = df[0] >= xlim[0]
    df = df.loc[mask1 & mask2]

    mask2 = df[1] <= ylim[1]
    mask1 = df[1] >= ylim[0]
    df = df.loc[mask1 & mask2]
    df = df.dropna()
    return df


def dtm_clip_buffer(df, buffer):
    pt = gpd.GeoSeries(gpd.points_from_xy(df[0], df[1], df[2]))
    mask = buffer.contains(pt)

    df_clipped = pt[mask == True]
    df_clipped = df_clipped.get_coordinates(include_z=True)
    return df_clipped


def dtm_clip(x, y, dtm):
    xlim = [x.min(), x.max()]
    ylim = [y.min(), y.max()]

    df = tif_clip_line_lim(xlim, ylim, dtm)
    buffer = dtm_get_buffer(x, y)
    df = dtm_clip_buffer(df, buffer)
    return df


def dtm_sample(x, y, dtm, tolerance=1, srtm=False):
    if srtm == True:
        tolerance = 31

    df_r = dtm_clip(x, y, dtm)

    pt1 = gpd.points_from_xy(x, y)
    pt1 = [MultiPoint([g]) for g in pt1]

    pt2 = gpd.points_from_xy(df_r["x"], df_r["y"], df_r["z"])
    pt2 = MultiPoint(pt2)

    pt1, pt3 = nearest_points(pt1, pt2)

    dist = shapely.distance(pt1, pt3)

    coords = shapely.get_coordinates(pt3)
    df_l = pd.DataFrame(np.column_stack([coords, dist]), columns=["x", "y", "dtm_dist"])

    df_l.index = df_l["x"].astype(str) + df_l["y"].astype(str)
    df_r.index = df_r["x"].astype(str) + df_r["y"].astype(str)

    df_dtm = pd.merge(df_l, df_r["z"], "left", left_index=True, right_index=True)
    df_dtm = df_dtm.rename(columns={"z": "z0_dtm"})

    sec_merge = np.column_stack([df_dtm.loc[:, "z0_dtm"], df_dtm.loc[:, "dtm_dist"]])
    sec_merge = np.round(sec_merge, 2)

    df_dtm = df_dtm.loc[df_dtm["dtm_dist"] <= tolerance]
    df_dtm = df_dtm.reset_index(drop=True)

    return sec_merge, df_dtm


def dtm_merge_data(ert):

    coords = ert.sec.values
    coords = coords[:, [0, 10, 9]]

    coords_int = coords_interpolate(coords)
    coords_int[:, 1:] = np.round(coords_int[:, 1:], 2)

    df_l = ert.data.copy()
    df_r = coords_int
    df_r = pd.DataFrame(df_r[:, 1:], index=df_r[:, 0], columns=["z0_dtm", "dtm_dist"])

    df = pd.merge(df_l, df_r, "left", left_on="ld", right_index=True)

    return df


def dtm_check(ert):
    check = False
    if np.sum(ert.sec["y"], axis=0) == 0:
        print("\t Placeholder coordinates only! Skipping DTM load..")
    else:
        check = True

    return check


def dtm_clip_3d(dtm, data):

    xlim = [data["x"].min(), data["x"].max()]
    ylim = [data["y"].min(), data["y"].max()]

    df_dtm = tif_clip_line_lim(xlim, ylim, dtm)
    return df_dtm
