from ares_py.ares_2dm.read_2dm import load_2dm
from ares_py.tools.colors import load_clr_scale
import numpy as np


class ERT:
    def __init__(self, fp):
        self.fp_load = fp
        pass

    def Load(self, cs_res="cs_def"):
        self.cs_res = dict(name=cs_res)

        if self.fp_load.endswith(".2dm"):
            self = load_2dm(self)
        return self

    def Colorscales(self):
        self.cs_res = load_clr_scale(self.cs_res)
        return self
