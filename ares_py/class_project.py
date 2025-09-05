import pandas as pd
import os
import geopandas as gpd

from ares_py.class_ert import ERT
from ares_py.get_ld import get_ld
from ares_py.tools.geometry_tools import pt_to_ls
from ares_py.layouts.layout import get_xy_ranges, get_extents, get_qc_grph_ls
from ares_py.layouts.topo import export_topo_ls, export_topo_pt


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

    def Process_2dm(self, inv_zond=True):
        data = []
        sec = []
        coords_int = []
        grd_mask = []
        ert_lines = {}

        for fp in self.fp["in_2dm"]:
            print(fp)
            ert = ERT(fp).Process_2dm(inv_zond=inv_zond)
            data.append(ert.data)
            sec.append(ert.sec)
            ert_lines[ert.line] = ert

            coords_tmp = pd.DataFrame(ert.coords_int)
            coords_int.append(coords_tmp)

            grd_mask.append(ert.grd_mask)

        self.data = pd.concat(data).reset_index(drop=True)
        self.sec = pd.concat(sec).reset_index(drop=True)
        self.coords_int = pd.concat(coords_int).reset_index(drop=True)
        self.sec["ID"] = self.sec["ID_line"].astype("Int64") * 10000
        self.sec["ID"] += self.sec["ld"]

        self.grd_mask = pd.concat(grd_mask).reset_index(drop=True)
        self.grd_mask.to_file(self.fp["ert"], layer="mask_grid", engine="pyogrio")

        self.ert = ert_lines

        return self

    def Export_gpkg_lines(self):
        gdf_ls = pt_to_ls(
            self.sec,
            x="x",
            y="y",
            order="ld",
            groupby="ID_line",
            crs=self.crs,
            z="z0",
        )

        gdf_ls.to_file(self.fp["ert"], layer="ert_sec_ls", engine="pyogrio")

        geom = gpd.points_from_xy(x=self.sec["x"], y=self.sec["y"], z=self.sec["z0"])
        gdf_pt = gpd.GeoDataFrame(self.sec, geometry=geom, crs=self.crs)
        gdf_pt.to_file(self.fp["ert"], layer="ert_sec_pt", engine="pyogrio")
        return self

    def Proc_layout_qc(self, scale, graph_height, res_max):

        scale = scale / 1000
        self.layout_qc = {}
        self.layout_qc["scale"] = scale
        self.layout_qc["graph_height"] = graph_height
        self.layout_qc["res_max"] = res_max

        lines = self.sec["ID_line"].unique()

        xyrange = get_xy_ranges(self.data, self.sec, scale, graph_height)
        cols = ["min", "max"]
        self.layout_qc["xrange"] = pd.DataFrame(xyrange[0], index=lines, columns=cols)
        self.layout_qc["yrange"] = pd.DataFrame(xyrange[1], index=lines, columns=cols)

        self.layout_qc["extents"] = get_extents(lines, xyrange[0], xyrange[1], scale)
        self.layout_qc["extents"]["res_max"] = res_max
        layer = "ert2d_lqc_extents_pl"

        self.layout_qc["extents"].to_file(self.fp["ert"], layer=layer, engine="pyogrio")

        self.layout_qc["qc_grph"] = get_qc_grph_ls(self)

        export_topo_ls(self)
        export_topo_pt(self)

        return self
