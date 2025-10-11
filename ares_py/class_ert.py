import numpy as np
from pathlib import Path
import os
import pandas as pd

from ares_py.read_2dm import load_2dm
from ares_py.inv_r2d import export_to_res2d
from ares_py.sections import get_sections
from ares_py.coords import coords_interpolate, create_csv_flat


class ERT:
    def __init__(self, fp, project):
        self.fp_load = fp
        self.project = project
        self.line = Path(self.fp_load).stem
        self.cs_res = None
        self.sec = None
        self.check = dict(dtm=False)
        self.Get_fps()

    def __call__(self):
        return pd.DataFrame(self.data)

    def Load(self, cs_res="cs_def"):
        self.cs_res = dict(name=cs_res)

        if self.fp_load.endswith(".2dm"):
            self = load_2dm(self)
            self.sec = get_sections(self)
        return self

    def Add_coordinates(self, mode):
        create_csv_flat(self.data, self.fps["crd_csv"]["flat"])
        fp = self.fps["crd_csv"][mode]

        if not os.path.exists(fp):
            print(f"{fp}\t not found! Falling back to flat coords...")
            fp = self.fps["crd_csv"]["flat"]
        # fp = str(fp).replace(".csv", f"_{mode}.csv")

        crd = pd.read_csv(fp)
        crd = crd.set_index("ld")
        cols_crd = [col for col in crd.columns if col.endswith(f"_{mode}")]
        crd = crd[cols_crd]
        crd.columns = crd.columns.str.rstrip(f"_{mode}")
        cols_crd = crd.columns
        crd = crd.reset_index()

        crd = coords_interpolate(crd["ld"], crd.iloc[:, 1:])
        crd = crd.round(2)
        crd = pd.DataFrame(crd[:, 1:], index=crd[:, 0])
        crd.columns = cols_crd
        cols_data = [col for col in self.data.columns if not col in crd.columns]
        self.data = self.data[cols_data]

        self.data = pd.merge(self.data, crd, how="left", left_on="ld", right_index=True)
        self.data["z"] = self.data["z0"] + self.data["doi"]
        return self

    def Process_2dm(self, crd_mode):
        self.crd_mode = crd_mode
        self.Load()
        self.Add_coordinates(crd_mode)
        self.Get_topo()
        self.Export_inversion()

        return self

    def Get_fps(self):
        self.fps = {}
        self.fps["in_topo"] = Path(self.project.fps["crd"], f"{self.line}.csv")
        self.fps["dtm"] = self.project.fps["dtm"]
        self.fps["ert"] = self.project.fps["ert"]
        self.fps["r2d_dat"] = Path(
            self.project.fps["inv"], f"{self.line}_r2d_input.dat"
        )

        self.fps["crd_csv"] = {}
        for mode in ["flat", "rec", "man", "plan"]:
            fn = str(self.line).zfill(3) + f"_{mode}.csv"
            self.fps["crd_csv"][mode] = Path(self.project.fps["crd"], fn)

        return self

    def Export_inversion(self):
        export_to_res2d(self)
        return self

    def Get_topo(self):

        fp = str(self.fps["in_topo"]).replace(".csv", f"_{self.crd_mode}.csv")
        df = pd.read_csv(fp)
        df = df.loc[df["ld"] % self.el_space == 0]

        last_electrode = max(self.data.iloc[:, :4].max())
        df = df.loc[df["ld"] <= last_electrode]
        df.columns = df.columns.str.rstrip(f"_{self.crd_mode}")
        df["ID_line"] = int(self.line)
        self.topo = df.reset_index(drop=True)
        return self
