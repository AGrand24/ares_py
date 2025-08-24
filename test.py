import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ares_py.get_ld import get_ld
from ares_py.class_ert import ERT
from ares_py.plot.fig import fig_meas_data
from ares_py.geometry.coords import (
    coords_load,
    coords_ld2d,
    coords_interpolate,
    coord_merge,
    coords_merge_sections,
    coords_get_z,
)
from ares_py.geometry.dtm import tif_read, dtm_sample, dtm_merge_data

fps = get_ld("input", ext=".2dm")["fp"].iloc[:4]
dtm = tif_read("input/klinovec/dtm.tif")

for fp in fps:
    print(fp)
    ert = ERT(fp).Load().Colorscales()

    try:
        coords = coords_load(ert.fp_load.replace(".2dm", ".csv"))
        ert.coords = coords_ld2d(coords)

        ert.coords_int = coords_interpolate(ert.coords)
        ert.data = coord_merge(ert)
        ert.sec = coords_merge_sections(ert)
        ert.sec, ert.dtm = dtm_sample(ert, dtm)
        ert.data = dtm_merge_data(ert)
        ert.data = coords_get_z(ert)
    except:
        print("\tError reading topo data..")
    fig = fig_meas_data(ert)
