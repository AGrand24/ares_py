import pandas as pd

pd.options.plotting.backend = "plotly"
import numpy as np
import plotly.graph_objects as go


from ares_py.class_ert import ERT
from ares_py.class_project import Project
from ares_py.get_ld import get_ld

prj = Project("vlkanova", crs=8353)

prj = prj.Process_2dm()
prj = prj.Export_gpkg_lines()
prj = prj.Proc_layout_qc(500, 130, 200)
