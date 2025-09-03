import pandas as pd
import os

from ares_py.class_ert import ERT
from ares_py.get_ld import get_ld


class Project:
    def __init__(self, name, crs=3857):
        self.name = name
        self.crs = crs
        self.Get_fp()

    def Get_fp(self):
        self.dir_in = f"input/{self.name}"
        self.dir_out = f"output/{self.name}"

        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)

        self.fp = dict(
            dtm=self.dir_in + "/dtm.tif",
            ert=self.dir_out + "/ert.gpkg",
        )
        self.fp["in_2dm"] = get_ld(self.dir_in, ext=".2dm")["fp"]
        return self

    def Process_2dm(self):
        data = []
        sec = []
        coords_int = []
        ert_lines = {}

        for fp in self.fp["in_2dm"]:
            print(fp)
            ert = ERT(fp).Process_2dm()
            data.append(ert.data)
            sec.append(ert.sec)
            coords_int.append(ert.coords_int)
            ert_lines[ert.line] = ert

        self.data = pd.concat(data).reset_index(drop=True)
        self.sec = pd.concat(sec).reset_index(drop=True)
        self.coords_int = pd.concat(coords_int).reset_index(drop=True)
        self.sec["ID"] = self.sec["ID_line"].astype("Int64") * 10000
        self.sec["ID"] += self.sec["ld"]
        self.ert = ert_lines

        return self
