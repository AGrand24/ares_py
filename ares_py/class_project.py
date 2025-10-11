import pandas as pd
import numpy as np
import os
import shutil
import geopandas as gpd
import shapely
from pathlib import Path
from shapely import MultiPoint


from ares_py.class_ert import ERT
from ares_py.tools.get_ld import get_ld
from ares_py.tools.geometry_tools import pt_to_ls
from ares_py.tools.colors import get_ref_res

# from ares_py.layouts.layout import get_xy_ranges, get_extents, get_qc_grph_ls
from ares_py.coords import (
    process_line_distance,
    export_crd_csv,
    calc_line_distance,
    coords_interpolate,
    export_crd_man,
    export_topo_ls,
    export_topo_pt,
)
from ares_py.sections import ld2sec
from ares_py.agrid import ag_kriging, ag_export_surfer
from ares_py.qc_lines import get_levels, fig_qc_lines


class Project:
    def __init__(self, name, crs=3857, res_range=[1, 10000]):
        self.name = name
        self.crs = crs
        self.Get_fps()
        self.Create_structure()
        self.res_range = res_range

    def Get_fps(self):
        root = Path("projects/", f"{self.name}/")
        self.fps = dict(root=root)
        self.fps["input"] = Path(root, "01_input/")
        self.fps["crd"] = Path(root, "02_coordinates/")
        self.fps["inv"] = Path(root, "03_inversion/")
        self.fps["grd"] = Path(root, "04_grd/")
        self.fps["qgis"] = Path(root, "qgis/")

        # self.fps["output"] = Path(root, "output/")
        self.folders = self.fps.copy()

        self.fps["dtm"] = Path(self.fps["qgis"], "dtm.tif")
        self.fps["ert"] = Path(self.fps["qgis"], "ert.gpkg")
        self.fps["rec"] = Path(self.fps["qgis"], "rec.gpkg")

        self.fps["in_2dm"] = get_ld(self.fps["input"], ext=".2dm")["fp"]

        self.fps["qgis_prj_spatial"] = Path(
            self.fps["qgis"], f"ert_{self.name}_cloud.qgs"
        )
        self.fps["qgis_prj_2d"] = Path(root, f"ert_{self.name}_2d.qgs")
        return self

    def Create_structure(self):
        for fld in self.folders.values():
            if not os.path.exists(fld):
                os.makedirs(fld)

        for key in ["qgis_prj_spatial", "qgis_prj_2d"]:
            fp = self.fps[key]
            if not os.path.exists(fp):
                fn = str(fp.stem).replace(f"ert_{self.name}", "ert_template") + ".qgs"
                fp_template = Path("ares_py/qgis/", fn)
                shutil.copy(fp_template, fp)

    def Process_2dm(self, crd_mode):
        data = []
        topo = []
        ert_lines = {}

        for fp in self.fps["in_2dm"]:
            print(fp)
            ert = ERT(fp, self).Process_2dm(crd_mode)
            data.append(ert.data)
            topo.append(ert.topo)
            ert_lines[ert.line] = ert

        self.data = pd.concat(data).reset_index(drop=True)
        self.topo = pd.concat(topo).reset_index(drop=True)
        self.topo["ID"] = self.topo["ID_line"].astype("Int64") * 10000
        self.topo["ID"] += self.topo["ld"]

        self.ert = ert_lines

        self.Export_gpkg_2dm()

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

        gdf_ls.to_file(self.fps["ert"], layer="ert_sec_ls", engine="pyogrio")

        geom = gpd.points_from_xy(x=self.sec["x"], y=self.sec["y"], z=self.sec["z0"])
        gdf_pt = gpd.GeoDataFrame(self.sec, geometry=geom, crs=self.crs)
        gdf_pt.to_file(self.fps["ert"], layer="ert_sec_pt", engine="pyogrio")
        return self

    def Proc_topo_2d(self):
        export_topo_ls(self)
        export_topo_pt(self)
        return self

    def Process_crd_plan(self):
        fp_in = "tmp/tmp_plan.gpkg"
        if os.path.exists(fp_in):
            gdf = gpd.read_file(fp_in)
            gdf = gdf.to_crs(self.crs)

            mask = gdf["ld_hor"] % 1 == 0
            gdf = gdf.loc[mask]

            gdf = process_line_distance(gdf)

            gdf["ID_line"] = gdf["ID_line"].astype(int)
            gdf["ID"] = gdf["ID_line"] * 10000 + gdf["ld"]

            gdf[["el", "sec", "el_sec"]] = ld2sec(gdf["ld"], gdf["el_space"])

            geom = gpd.points_from_xy(gdf["x"], gdf["y"], gdf["z0"])
            gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=self.crs)

            gdf.to_file(self.fps["ert"], layer="ert_plan_pt", engine="pyogrio")

            self.crd_plan = gdf

            self.Process_crd_csv_plan()
        else:
            print(f"Missing...{fp_in}\nRun ert_ls2pt model in QGIS")

        return self

    def Process_crd_csv_plan(self):
        for line in self.crd_plan["ID_line"].unique():
            df = self.crd_plan.loc[self.crd_plan["ID_line"] == line]

            df = df[["ld", "ld_hor", "x", "y", "z0"]].set_index("ld")
            df = df.add_suffix("_plan")
            export_crd_csv(df, line, self)

    def Get_el_space(self):
        lines = []
        el_space = []

        for ert in self.ert.values():
            lines.append(int(ert.line))
            el_space.append(int(ert.el_space))

        df = pd.DataFrame({"ID_line": lines, "el_space": el_space})
        return df

    def Process_crd_rec(self):

        gdf = gpd.read_file(self.fps["rec"], engine="pyogrio", layer="rec_ert_pt")
        gdf = gdf.sort_values(["ID_line", "ld"])

        for ert in self.ert.values():
            line = int(ert.line)

            tmp = gdf.loc[gdf["ID_line"] == line].copy()
            print(f"Processing rec crd - line {line}, {len(tmp)} entries")

            if len(tmp) > 0:
                tmp[["x", "y", "z0"]] = tmp.get_coordinates(include_z=True)
                tmp = tmp.set_index("ld")
                tmp = tmp[["x", "y", "z0"]]

                tmp["ld_hor"] = calc_line_distance(tmp["x"], tmp["y"])
                tmp = tmp.round(2)
            else:
                tmp = pd.DataFrame(columns=["ld_hor", "x", "y", "z0"])

            tmp = tmp.add_suffix("_rec")
            tmp.to_csv(str(ert.fps["in_topo"]).replace(".csv", "_rec.csv"), index=False)

        return self

    def Process_crd_man(self):

        layers = gpd.list_layers(self.fps["ert"])
        layers = layers["name"].values

        if not "crd_man_input_pt" in layers:
            gdf = gpd.read_file(self.fps["rec"], layer="rec_ert_pt")
            gdf.to_file(self.fps["ert"], layer="crd_man_input_pt")

        gdf = gpd.read_file("tmp/tmp_rec.gpkg")
        gdf_out = []
        for line in gdf["ID_line"].unique():
            tmp = gdf.loc[gdf["ID_line"] == line].copy()
            tmp[["x", "y", "z0"]] = tmp.get_coordinates(include_z=True)
            tmp["ld"] = calc_line_distance(tmp["x"], tmp["y"], tmp["z0"])
            tmp = coords_interpolate(
                tmp["ld"], tmp[["ld_hor", "ID_line", "x", "y", "z0"]]
            )
            tmp = gpd.GeoDataFrame(tmp)
            tmp.columns = ["ld", "ld_hor", "ID_line", "x", "y", "z0"]
            gdf_out.append(tmp)

        gdf_out = pd.concat(gdf_out).reset_index(drop=True)
        gdf_out = gdf_out.set_geometry(
            gpd.points_from_xy(gdf_out["x"], gdf_out["y"], gdf_out["z0"])
        )
        gdf_out = gdf_out.set_crs(self.crs)
        gdf_out = gdf_out.loc[gdf_out["ld"] % 1 == 0]

        df = self.Get_el_space()
        df = df.set_index("ID_line")

        gdf_out = pd.merge(gdf_out, df, "left", left_on="ID_line", right_index=True)

        gdf_out.to_file(self.fps["ert"], layer="crd_man_proc_pt", engine="pyogrio")

        gdf_out[["x", "y"]] = gdf_out.get_coordinates()
        gdf_ls = pt_to_ls(gdf_out, x="x", y="y", groupby="ID_line", order="ld")
        gdf_ls.to_file(self.fps["ert"], layer="crd_man_proc_ls")

        export_crd_man(self)
        return self

    def Grid_data(self, skip=None, cell_size=1):
        ld = get_ld(self.fps["grd"], ext=".tsv")

        for fp, fn in zip(ld["fp"], ld["f"]):
            try:
                line = int(fn[:3])
                df = pd.read_csv(fp, sep="\t")
                df["x"] = df["x"] % 10000
                df["x"] += line * 10000

                if skip != None and line not in skip:
                    print(f"\nGridding:\t {fp}")
                    data = [df["x"], df["z"], df["res"]]
                    data[2] = np.log10(data[2])
                    grd = ag_kriging(data, cell_size=cell_size, exact=True)
                    grd[2] = 10 ** grd[2]
                    fp_out = str(fp).replace(".tsv", ".grd")
                    ag_export_surfer(grd, 2, fp_out)
            except:
                print(f"\nError- {fp}\n")

        return self

    def Grid_mask(self):

        geom = []
        lines = []
        for ert in self.ert.values():
            line = int(ert.line)
            data = ert.data
            x = ert.data["ld_hor"] + 10000 * line
            y = ert.data["z"]

            points = MultiPoint(list(zip(x, y)))
            polygon = shapely.concave_hull(points, 0.03)
            polygon = shapely.buffer(polygon, 0.25)
            geom.append(polygon)
            lines.append(line)

        gdf = gpd.GeoDataFrame(data={"ID_line": lines}, geometry=geom, crs=self.crs)
        gdf.to_file(self.fps["ert"], layer="ert2d_grd_mask_pl", engine="pyogrio")
        return self

    def Merge_contours(self, type):
        ld = get_ld("tmp/")
        ld = ld.loc[ld["fn"].str.startswith("tmp_cnt")]
        print(ld)
        gdf = []
        for fp in ld["fp"]:
            gdf.append(gpd.read_file(fp))

        gdf = pd.concat(gdf)
        gdf = gdf.loc[gdf["ELEV"] % 50 == 0]
        gdf.crs = 8353
        gdf.to_file(self.fps["ert"], layer=f"cnt_{type}_full_ls")
        return self

    def Atlas_qc(self, atlas_input):

        xrange = []
        zmax = []
        for i, ert in enumerate(self.ert.values()):
            xrange.append(ert.topo["ld_hor"].max() - ert.topo["ld_hor"].min())
            zmax.append(ert.data["z0"].max())

        atlas_input["xrange"] = xrange
        atlas_input["zmax"] = zmax

        atlas = pd.DataFrame(atlas_input)
        atlas = pd.concat([atlas, self.Get_el_space()], axis=1).set_index("ID_line")
        atlas["xmin"] = 10000 * atlas.index
        atlas["xmax"] = atlas["xmin"] + xrange
        atlas["zmax"] = 5 * (atlas["zmax"] // 5) + atlas["z_gap"]
        atlas["map_height"] = (atlas["page_heights"] - 30) / 3

        atlas["zmin"] = atlas["zmax"] - atlas["map_height"] * atlas["map_scales"] / 1000
        atlas["zrange"] = atlas["zmax"] - atlas["zmin"]
        atlas["page_width"] = atlas["xrange"] * (1000 / atlas["map_scales"]) + 25 + 10

        from shapely import Polygon

        geom = []
        crds = zip(atlas["xmin"], atlas["xmax"], atlas["zmin"], atlas["zmax"])
        for x1, x2, y1, y2 in crds:
            geom.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

        atlas = gpd.GeoDataFrame(atlas, geometry=geom, crs=self.crs)
        atlas.to_file(self.fps["ert"], engine="pyogrio", layer="atlas_qc")

        self.atlas_qc = atlas
        return self

    def QC_lines(self):

        for ert in self.ert.values():
            line = int(ert.line)
            ad = self.atlas_qc.copy().loc[line, :]
            df = ert.data.copy()
            levels = get_levels(df, ad["lvl_frequency"])
            self.fig_qc_lines = fig_qc_lines(self, levels, ad, line)

        return self

    def Export_gridding_input(self):

        ld = get_ld(self.fps["inv"], ext=".dat")

        mask = (ld["fn"].str.contains("_znd")) | (
            ld["fn"].str.contains("_topres.dat")
            | (ld["fn"].str.contains("_topreslog.dat"))
        )
        ld = ld.loc[mask]

        for fp in ld["fp"]:
            fp_out = Path(self.fps["grd"], Path(fp).stem + ".tsv")
            fp_out = str(fp_out).replace("_topres.tsv", "_r2d.tsv")
            fp_out = str(fp_out).replace("_topreslog.tsv", "_r2d.tsv")

            if "_topres.dat" in fp:
                res_col = 2
                df = pd.read_fwf(fp, skiprows=1, header=None)
            if "_topreslog.dat" in fp:
                res_col = 2
                df = pd.read_fwf(fp, skiprows=1, header=None)
                df.iloc[:, 2] = 10 ** df.iloc[:, 2]
            else:
                res_col = 3
                df = pd.read_csv(fp, sep="\t", skiprows=1, header=None)

            df = df.iloc[:, [0, 1, res_col, res_col]]
            df.iloc[:, 3] = np.log10(df.iloc[:, 2])
            df.columns = ["x", "z", "res", "log_res"]
            df.to_csv(fp_out, sep="\t", index=False)
            print(f"Exported...{fp_out}")

        return self

    def Export_gpkg_2dm(self):
        gdf = self.data.copy()

        x = gdf["ld_hor"] + (gdf["ID_line"] * 10000)
        geom = gpd.points_from_xy(x, gdf["z"])
        gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=self.crs)
        gdf = gdf.reset_index(drop=True)

        gdf["ref_res"] = get_ref_res(gdf["res"], self.res_range)

        gdf["ref_std"] = gdf["std"]
        gdf.loc[0, "ref_std"] = 0
        gdf.loc[1, "ref_std"] = 10
        gdf["ref_std"] = np.clip(gdf["ref_std"], 0, 10)

        gdf["ref_ep"] = gdf["ep"]
        gdf.loc[0, "ref_ep"] = -100
        gdf.loc[1, "ref_ep"] = 100
        gdf["ref_ep"] = np.clip(gdf["ref_ep"], -100, 100)

        df = self.Get_el_space().set_index("ID_line")
        gdf = pd.merge(gdf, df, "left", left_on="ID_line", right_index=True)
        gdf.to_file(self.fps["ert"], layer="ert2d_2dm_pt", engine="pyogrio")

    def Merge_el_space(self, df_in):
        df = self.Get_el_space()
        df = df.set_index("ID_line")

        df_out = pd.merge(df_in, df, how="left", left_on="ID_line", right_index=True)
        return df_out

    def Export_gpkg_inv(self):

        ld = get_ld(self.fps["grd"], ext=".tsv")
        ld["type"] = ld["f"].str.slice(4, None)
        ld["ID_line"] = ld["f"].str.slice(None, 3).astype(int)

        for type in ld["type"].unique():
            ld_type = ld.copy()
            ld_type = ld_type.loc[ld_type["type"] == type]

            gdf = []
            for fp, line in zip(ld_type["fp"], ld_type["ID_line"]):
                df = pd.read_csv(fp, sep="\t")
                df["ID_line"] = line
                gdf.append(df)

            gdf = pd.concat(gdf).reset_index(drop=True)
            gdf.columns = ["ld_hor", "z", "res", "res_log", "ID_line"]
            gdf["ld_hor"] += gdf["ID_line"] * 10000
            geom = gpd.points_from_xy(gdf["ld_hor"], gdf["z"])
            gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=self.crs)

            gdf["ref_res"] = get_ref_res(gdf["res"], self.res_range)
            gdf = self.Merge_el_space(gdf)

            gdf.to_file(self.fps["ert"], layer=f"inv_{type}_pt", engine="pyogrio")

        return self
