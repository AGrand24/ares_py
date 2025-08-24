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
    coords_create_csv,
)
from ares_py.geometry.dtm import tif_read, dtm_sample, dtm_merge_data, dtm_check

fps = get_ld("input", ext=".2dm")["fp"]
dtm = tif_read("input/klinovec/dtm.tif")

for fp in fps:
    print(fp)
    ert = ERT(fp).Load().Colorscales()

    coords_create_csv(ert)
    coords = coords_load(ert.fp["in_topo"])
    ert.coords = coords_ld2d(coords)
    ert.coords_int = coords_interpolate(ert.coords)
    ert.data = coord_merge(ert)
    ert.sec = coords_merge_sections(ert)

    ert.check["dtm"] = dtm_check(ert)

    if ert.check["dtm"] == True:

        ert.sec, ert.dtm = dtm_sample(ert, dtm)
        ert.data = dtm_merge_data(ert)
        zmode = "dtm"
    else:
        zmode = "topo"

    ert.data = coords_get_z(ert, zmode)
    fig = fig_meas_data(ert)
