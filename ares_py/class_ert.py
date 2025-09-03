import numpy as np
from pathlib import Path
import os

from ares_py.ares_2dm.read_2dm import load_2dm
from ares_py.tools.colors import load_clr_scale
from ares_py.geometry.sections import get_sections

import numpy as np

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
from ares_py.geometry.dtm import dtm_sample, dtm_merge_data, dtm_check, tif_read


class ERT:
    def __init__(self, fp):
        self.fp_load = fp
        self.project = Path(self.fp_load).parents[0].name
        self.line = Path(self.fp_load).stem
        self.cs_res = None
        self.sec = None
        self.check = dict(dtm=False)
        self.Get_fp()

    def Load(self, cs_res="cs_def"):
        self.cs_res = dict(name=cs_res)

        if self.fp_load.endswith(".2dm"):
            self = load_2dm(self)
            self.sec = get_sections(self)
        return self

    def Process_2dm(self):
        self.Load()
        self.Colorscales()

        coords_create_csv(self)
        coords = coords_load(self.fp["in_topo"])
        self.coords = coords_ld2d(coords)
        self.coords_int = coords_interpolate(self.coords)
        self.data = coord_merge(self)
        self.sec = coords_merge_sections(self)

        self.check["dtm"] = dtm_check(self)

        if self.check["dtm"] == True:
            self.dtm = tif_read(self.fp["dtm"])
            self.sec, self.dtm = dtm_sample(self, self.dtm)
            self.data = dtm_merge_data(self)
            zmode = "dtm"
        else:
            zmode = "topo"

        print("\t\t\t", zmode)
        self.data = coords_get_z(self.data, "data", zmode=zmode)
        self.sec = coords_get_z(self.sec, "sec", zmode=zmode)
        self.fig2d = fig_meas_data(self)

        self.data["ID_line"] = self.line
        self.sec["ID_line"] = self.line
        return self

    def Get_fp(self):
        self.fp = dict(
            name=Path(self.fp_load).stem,
            out_dir=Path("output", self.project),
            in_dir=Path(self.fp_load).parents[0],
        )
        self.fp["in_topo"] = self.fp_load.replace(".2dm", ".csv")
        self.fp["dtm"] = os.path.join(self.fp["in_dir"], "dtm.tif")

        for ext in ["html"]:
            out_dir = self.fp["out_dir"]
            name = self.fp["name"]
            self.fp[f"out_{ext}"] = Path(out_dir, name + ".html")

        return self

    def Colorscales(self):
        self.cs_res = load_clr_scale(self.cs_res)
        return self
