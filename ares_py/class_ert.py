from ares_py.ares_2dm.read_2dm import load_2dm
from ares_py.tools.colors import load_clr_scale
from ares_py.geometry.sections import get_sections
import numpy as np


class ERT:
    def __init__(self, fp):
        self.fp_load = fp
        self.cs_res = None
        self.sec = None

    def Load(self, cs_res="cs_def"):
        self.cs_res = dict(name=cs_res)

        if self.fp_load.endswith(".2dm"):
            self = load_2dm(self)
            self.sec = get_sections(self)
        return self

    def Colorscales(self):
        self.cs_res = load_clr_scale(self.cs_res)
        return self
