import os
import numpy as np
import geopandas as gpd
import pandas as pd

from ares_py.tools.agrid import ag_kriging, ag_export_surfer, ag_pt2pl, ag_mask, ag_plot


def parse_grd_data(df, line):
    df.iloc[:, 0] += int(line) * 10000

    grd_data = [0, 1, 2]
    grd_data[0] = df.iloc[:, 0].values
    grd_data[1] = df.iloc[:, 1].values
    grd_data[2] = df.iloc[:, 2].values
    return grd_data


def export_grd(grd_data, fp):
    grd = ag_kriging(grd_data, cell_size=0.5, exact=True)
    grd[2] = np.power(10, grd[2])
    ag_export_surfer(grd, 2, fp)


def get_grd_mask(ert):
    df = ert.data
    line = ert.line

    mask_x = df["ld_hor"].values + int(line) * 10000
    poly_mask = ag_pt2pl([mask_x, df["z"]], buffer=0, ratio=0.1)
    gdf = gpd.GeoDataFrame(geometry=[poly_mask], crs=8353)
    gdf["ID_line"] = line
    return gdf


def export_inversion_zond(ert):
    fp = ert.fp["inv_zond"]

    if os.path.exists(fp):
        df = pd.read_csv(fp, sep="\t", skiprows=1, header=None)
        grd_data = parse_grd_data(df, ert.line)
        export_grd(grd_data, ert.fp["grd_zond"])

    else:
        print("\tZond inversion file not found!")
