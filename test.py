from ares_py.class_ert import ERT
from ares_py.plot.fig import fig_meas_data
import numpy as np

ert = ERT("input/klinovec/001.2dm").Load().Colorscales()

fig_meas_data(ert)
