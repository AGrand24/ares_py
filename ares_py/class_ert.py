import numpy as np
from pathlib import Path

from ares_py.ares_2dm.read_2dm import load_2dm
from ares_py.tools.colors import load_clr_scale
from ares_py.geometry.sections import get_sections


class ERT:
    def __init__(self, fp):
        self.fp_load = fp
        self.project = Path(self.fp_load).parents[0].name
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

    def Get_fp(self):
        self.fp = dict(
            name=Path(self.fp_load).stem,
            out_dir=Path("output", self.project),
            in_dir=Path(self.fp_load).parents[0],
        )
        self.fp["in_topo"] = self.fp_load.replace(".2dm", ".csv")
        for ext in ["html"]:
            out_dir = self.fp["out_dir"]
            name = self.fp["name"]
            self.fp[f"out_{ext}"] = Path(out_dir, name + ".html")

        return self

    def Colorscales(self):
        self.cs_res = load_clr_scale(self.cs_res)
        return self
