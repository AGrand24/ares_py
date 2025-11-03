import numpy as np
import pandas as pd
from ares_py.tools.geometry_tools import get_x2d


def get_electrodes(ert, section_electrodes=8):
    electrodes = ert.data.iloc[:, :4]
    el_max = np.max(electrodes)

    x = np.arange(start=0, stop=el_max + ert.el_space, step=ert.el_space).astype(int)
    xn = x / ert.el_space

    sec = xn // section_electrodes

    electrodes = np.column_stack([xn, sec, sec, xn * ert.el_space])
    electrodes[:, 2] = electrodes[:, 0] - electrodes[:, 1] * 8
    electrodes = electrodes.astype(int)

    electrodes = pd.DataFrame(electrodes)
    electrodes.columns = ["el", "sec", "el_sec", "ld"]
    electrodes["ld"] = electrodes["ld"].astype(float)
    electrodes["ld_hor"] = electrodes["ld"]
    electrodes["x"] = electrodes["ld"]
    electrodes["y"] = 0
    electrodes["z0"] = 0

    electrodes["x2d"] = get_x2d(electrodes["ld_hor"], ert.line)
    electrodes["ID_line"] = ert.line
    electrodes["ID"] = electrodes["ID_line"] * 10000 + electrodes["ld"]
    return electrodes


def ld2sec(ld, el_space, cable_el=8):
    el = ld / el_space
    sec = el // cable_el
    el_sec = el - (sec * cable_el)

    return np.column_stack([el, sec, el_sec])
