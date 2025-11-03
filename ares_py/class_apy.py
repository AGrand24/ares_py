import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os

from ares_py.coords import coords_interpolate
from ares_py.tools.geometry_tools import get_x2d


class Apy:
    def __init__(self, fp):
        self.fp_data = fp
        self.fp_el = Path(str(fp).replace(".apd", ".ape"))
        self.line = int(Path(fp).stem[:3])
        self.Load()

    def Load(self):
        self.data = pd.read_csv(self.fp_data, sep="\t")
        self.electrodes = pd.read_csv(self.fp_el, sep="\t")

        self.el_space = self.electrodes["ld"].iloc[1] - self.electrodes["ld"].iloc[0]
        self.el_space = int(self.el_space)
        self.data["el_space"] = self.el_space
        self.electrodes["el_space"] = self.el_space

        return self

    def Add_coordinates(self, fp):

        output = []
        for data in [self.data, self.electrodes]:
            if os.path.exists(fp):
                topo = pd.read_csv(fp).set_index("ld")
                cols_topo = topo.columns
                topo_interpolated = coords_interpolate(topo.index, values=topo)
                topo_interpolated = topo_interpolated.round(2)
                topo_interpolated = pd.DataFrame(
                    topo_interpolated[:, 1:],
                    index=topo_interpolated[:, 0],
                )
                topo_interpolated.columns = cols_topo
                cols_data = [col for col in data.columns if col not in cols_topo]
                data = data[cols_data]
                data = pd.merge(
                    data,
                    topo_interpolated,
                    how="left",
                    left_on="ld",
                    right_index=True,
                )
                data["x2d"] = get_x2d(data["ld_hor"], line=self.line)
                output.append(data)

            else:
                print(f"Coords not fount-\t {fp}")

        if len(output) > 1:
            self.data = output[0]
            self.electrodes = output[1]

        self.data["z"] = self.data["z0"] + self.data["doi"]
        return self

    def Save(self):
        self.data.to_csv(self.fp_data, sep="\t", index=False)
        self.electrodes.to_csv(self.fp_el, sep="\t", index=False)
        return self

    def Recalc_res(self):
        data = self.data.copy()
        elec = self.electrodes.copy()

        elec = elec[["ld", "x", "y", "z0"]]
        elec = elec.set_index("ld")

        x = []
        y = []
        z = []
        for i in list(range(4)):
            ld = data.copy().iloc[:, i]
            ld.name = "ld"
            xy = pd.merge(ld, elec, how="left", left_on="ld", right_index=True)
            x.append(xy["x"])
            y.append(xy["y"])
            z.append(xy["z0"])

        x = np.column_stack(x)
        y = np.column_stack(y)
        z = np.column_stack(z)

        indexes = [(0, 2), (1, 2), (0, 3), (1, 3)]

        d2d = []
        d3d = []
        for ind in indexes:
            i1 = ind[0]
            i2 = ind[1]

            dx = x[:, i2] - x[:, i1]
            dy = y[:, i2] - y[:, i1]
            dz = z[:, i2] - z[:, i1]

            d2d.append((dx**2 + dy**2) ** 0.5)
            d3d.append((dx**2 + dy**2 + dz**2) ** 0.5)

        d2d = 1 / np.column_stack(d2d)
        d3d = 1 / np.column_stack(d3d)

        k2d = 2 * np.pi / (d2d[:, 0] - d2d[:, 1] - d2d[:, 2] + d2d[:, 3])
        k3d = 2 * np.pi / (d3d[:, 0] - d3d[:, 1] - d3d[:, 2] + d3d[:, 3])

        data["k2d"] = k2d
        data["k3d"] = k3d

        resistance = data["v"] / data["i"]

        data["res2d"] = np.round(data["k2d"] * resistance)
        data["res3d"] = np.round(data["k3d"] * resistance)

        self.data = data
        return self
