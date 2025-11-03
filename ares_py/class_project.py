import pandas as pd
import numpy as np
import os
import shutil
import geopandas as gpd
import shapely
from pathlib import Path
from shapely import Polygon, MultiPolygon


from ares_py.class_raw import Raw
from ares_py.class_apy import Apy
from ares_py.tools.get_ld import get_ld
from ares_py.tools.geometry_tools import pt_to_ls
from ares_py.tools.colors import get_ref_res
from ares_py.r2d import dat_file_parse, dat_file_export


# from ares_py.layouts.layout import get_xy_ranges, get_extents, get_qc_grph_ls
from ares_py.coords import parse_ape
from ares_py.agrid import (
    run_krg,
    export_surfer,
    export_contours,
    read_surfer6_binary_grid,
)
from ares_py.qc_lines import get_levels, fig_qc_lines


class Project:
    def __init__(self, name, crs=8353, res_range=[1, 10000]):
        self.name = name
        self.crs = crs
        self.Get_fps()
        self.Create_structure()
        self.res_range = res_range

    def Get_fps(self):
        root = Path("projects/", f"{self.name}/")
        self.fps = dict(root=root)
        self.fps["doc"] = Path(root, "00_doc/")
        self.fps["2dm"] = Path(root, "01_2dm/")
        self.fps["apy"] = Path(root, "02_apy/")
        self.fps["crd"] = Path(root, "03_crd/")
        self.fps["qcl"] = Path(root, "04_qcl/")
        self.fps["inv"] = Path(root, "05_inv/")
        self.fps["grd"] = Path(root, "06_grd/")
        self.fps["mod"] = Path(root, "07_mod/")
        self.fps["qgis"] = Path(root, "99_qgs/")

        self.folders = self.fps.copy()

        self.fps["dtm"] = Path(self.fps["qgis"], "dtm.tif")
        self.fps["2d"] = Path(self.fps["root"], f"{self.name}_ert2d.gpkg")
        self.fps["spatial"] = Path(self.fps["qgis"], f"{self.name}_spatial.gpkg")
        self.fps["rec"] = Path(self.fps["qgis"], f"{self.name}_rec.gpkg")

        self.fps["in_2dm"] = get_ld(self.fps["2dm"], ext=".2dm")["fp"]

        self.fps["qgis_prj_spatial"] = Path(
            self.fps["qgis"], f"ert_{self.name}_cloud.qgs"
        )
        # self.fps["qgis_prj_2d"] = Path(root, f"ert_{self.name}_2d.qgs")
        return self

    def Create_structure(self):
        for fld in self.folders.values():
            if not os.path.exists(fld):
                os.makedirs(fld)

        for key in ["qgis_prj_spatial"]:
            fp = self.fps[key]
            if not os.path.exists(fp):
                fn = str(fp.stem).replace(f"ert_{self.name}", "ert_template") + ".qgs"
                fp_template = Path("ares_py/qgis/", fn)
                shutil.copy(fp_template, fp)

    def Process_raw(self):
        for fp in self.fps["in_2dm"]:
            print(fp)
            raw = Raw(fp).Load()
            raw = raw.Save(self.fps["apy"])
        return self

    def Process_apy(self):
        ld = get_ld(self.fps["apy"], ext="apd")

        keys = []
        vals1 = []
        vals2 = []
        for fp in ld["fp"]:
            print(fp)
            apy = Apy(fp).Load()
            fp_topo = Path(self.fps["crd"], f"{str(apy.line).zfill(3)}_topo.csv")
            apy = apy.Add_coordinates(fp_topo)
            apy = apy.Recalc_res()
            apy = apy.Save()
            vals1.append(apy)
            keys.append(apy.line)
            vals2.append(apy.el_space)

        self.apy = dict(zip(keys, vals1))
        self.el_space = dict(zip(keys, vals2))
        return self

    def Merge_apy_coordinates(self):
        ld = get_ld(self.fps["apy"], ext=".ape")

        electrodes = []
        coords = []
        for fp in ld["fp"]:
            data = parse_ape(fp)
            electrodes.append(data[0])
            coords.append(data[1])

        electrodes = pd.concat(electrodes).reset_index(drop=True)
        coords = pd.concat(coords).reset_index(drop=True)
        self.electrodes = electrodes
        self.coords = coords
        return self

    def Gpkg_coords(self):
        df = gpd.GeoDataFrame(self.coords)

        # Points
        geom_spatial = gpd.points_from_xy(df["x"], df["y"], df["z0"])
        geom_2d = gpd.points_from_xy(df["x2d"], df["z0"])

        gdf_sp = gpd.GeoDataFrame(df.copy(), geometry=geom_spatial, crs=self.crs)
        gdf_2d = gpd.GeoDataFrame(df.copy(), geometry=geom_2d, crs=self.crs)

        gdf_sp.to_file(self.fps["spatial"], layer=f"ert_pt")
        print(f"Exported - {self.fps["spatial"]} - ert_pt ")
        gdf_2d.to_file(self.fps["2d"], layer=f"topo_pt")
        print(f"Exported - {self.fps["2d"]} - topo_pt ")

        # LineStrings
        gdf_2d = pt_to_ls(
            gdf_2d, x="x2d", y="z0", order="ID", groupby="ID_line", crs=self.crs
        )
        gdf_sp = pt_to_ls(
            gdf_sp, x="x", y="y", order="ID", groupby="ID_line", crs=self.crs
        )

        gdf_sp.to_file(
            self.fps["spatial"],
            layer=f"ert_ls",
        )
        print(f"Exported - {self.fps["spatial"]} - ert_ls ")
        gdf_2d.to_file(
            self.fps["2d"],
            layer=f"topo_ls",
        )
        print(f"Exported - {self.fps["2d"]} - topo_ls ")
        return self

    def Gpkg_measured(self):
        ld = get_ld(self.fps["apy"], ext=".apd")

        gdf = []
        for fp in ld["fp"]:
            gdf.append(pd.read_csv(fp, sep="\t"))

        gdf = pd.concat(gdf).reset_index(drop=True)

        geom = gpd.points_from_xy(gdf["x2d"], gdf["z"])
        gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=self.crs)
        gdf = gdf.reset_index(drop=True)

        gdf["ref_res"] = get_ref_res(gdf["res"], self.res_range)

        gdf["ref_std"] = gdf["std"]
        gdf.loc[0, "ref_std"] = 0
        gdf.loc[1, "ref_std"] = 10
        gdf["ref_std"] = np.clip(gdf["ref_std"], 0, 10)

        gdf["ref_ep"] = gdf["ep"]
        gdf.loc[0, "ref_ep"] = -200
        gdf.loc[1, "ref_ep"] = 200
        gdf["ref_ep"] = np.clip(gdf["ref_ep"], -100, 100)

        print(f"\nExported - {self.fps['2d']} - measured_pt")
        gdf.to_file(self.fps["2d"], layer=f"measured_pt")

        return self

    def Gpkg_mask(self, manual_overwrite=False):

        ld = get_ld(self.fps["apy"], ext=".apd")

        geom = []
        line = []
        for fp_data in ld["fp"]:
            fp_el = str(fp_data).replace(".apd", ".ape")

            electrodes = pd.read_csv(fp_el, sep="\t")
            data = pd.read_csv(fp_data, sep="\t")
            line.append(data["ID_line"].iloc[0])

            electrodes = electrodes[["x2d", "z0"]].values

            data = data.groupby("x2d", as_index=False)["z"].agg("min")
            data["z"] = data["z"] - 1
            data = data.values
            data = data[::-1]

            data[-1, 0] = electrodes[0, 0]
            data[0, 0] = electrodes[-1, 0]

            pts = np.vstack([electrodes, data])
            geom.append(Polygon([(x, y) for x, y in zip(pts[:, 0], pts[:, 1])]))

        # geom = [MultiPolygon([pl for pl in geom])]
        gdf = gpd.GeoDataFrame(data={"ID_line": line}, geometry=geom, crs=self.crs)
        gdf = gdf.drop_duplicates(subset="ID_line", keep="first")

        fp = self.fps["2d"]
        gdf.to_file(fp, layer=f"mask_pl", engine="pyogrio")

        layers = gpd.list_layers(fp)["name"].values

        print(f"\nExported - {self.fps['2d']} - mask_pl")

        if not f"mask_man_pl" in layers or manual_overwrite == True:
            gdf.to_file(fp, layer=f"mask_man_pl", engine="pyogrio")
            print(f"Copied mask_pl to mask_man_pl!")

        return self

    def Export_r2d(self):
        print()
        ld = get_ld(self.fps["apy"], ext=".apd")
        for fp in ld["fp"]:
            line = str(Path(fp).stem)
            dat_lines = dat_file_parse(fp, self.name)

            fp_out = Path(self.fps["inv"], f"{line}_r2d_input.dat")
            dat_file_export(dat_lines, fp=fp_out)
            print(f"Exported - {fp_out}")
        return self

    def Export_gridding_input(self):

        ld = get_ld(self.fps["inv"], ext=".dat")

        mask = (ld["fn"].str.contains("_znd")) | (
            ld["fn"].str.contains("_topreslog.dat")
        )

        ld = ld.loc[mask]

        for fp in ld["fp"]:
            line = int(Path(fp).stem[:3])
            fp_out = Path(self.fps["grd"], Path(fp).stem + ".csv")
            # fp_out = str(fp_out).replace("_topres.csv", "_r2d.csv")
            fp_out = str(fp_out).replace("_topreslog.csv", "_r2d.csv")

            # if "_topres.dat" in fp:
            #     res_col = 2
            #     df = pd.read_fwf(fp, skiprows=1, header=None)
            if "_topreslog.dat" in fp:
                res_col = 2
                df = pd.read_fwf(fp, skiprows=1, header=None)
                df.iloc[:, 2] = 10 ** df.iloc[:, 2]
            else:
                res_col = 3
                df = pd.read_csv(fp, skiprows=1, header=None, sep="\t")
            df = df.iloc[:, [0, 1, res_col, res_col]]
            df.iloc[:, 3] = np.log10(df.iloc[:, 2])
            df.columns = ["x", "z", "res", "log_res"]
            df["res"] = df["res"].round(1)

            df = df.loc[np.abs(df["res"]) != np.inf]
            df["x2d"] = df["x"] + line * 10000

            df = df[["x2d", "z", "res", "log_res"]]

            df.to_csv(fp_out, index=False)
            print(f"Exported...{fp_out}")

        return self

    def Grid_data(self, skip=None, cell_size=1):
        grid_dict, ld = self.Get_grid_dictionary()

        for fp, fn, type, line in zip(ld["fp"], ld["f"], ld["type"], ld["ID_line"]):
            line = int(fn[:3])
            df = pd.read_csv(fp)
            try:
                if skip != None and line not in skip:
                    print(f"\nGridding:\t {fp}")
                    x = df["x2d"]
                    y = df["z"]
                    z = df["res"].clip(self.res_range[0], self.res_range[1])
                    z = np.log10(z)
                    grd = run_krg(
                        x=x,
                        y=y,
                        z=z,
                        cell_size=cell_size,
                        exact=True,
                        slope=1,
                        nugget=0.1,
                    )
                    grd[2] = 10 ** grd[2]
                    grid_dict[type][line] = grd
                    self.grids = grid_dict
                    print("Exporting surfer grid..")
                    fp_out = str(fp).replace(".csv", ".grd")
                    export_surfer(grd, 2, fp_out)
            except:
                print(f"\nError- {fp}\n")

        return self

    def Gpkg_contours(self, interval=10):
        ld = get_ld(self.fps["grd"], ext=".grd")
        ld["inv_type"] = ld["f"].str.slice(4, None)
        for type in ld["inv_type"].unique():
            ld_type = ld.copy(deep=True)
            ld_type = ld_type.loc[ld_type["inv_type"] == type]

            cnt = []
            for fp in ld_type["fp"]:
                print(f"Exporting contours - {fp}")
                # try:
                grid, header = read_surfer6_binary_grid(fp)
                x = np.array([header["xlo"], header["xhi"]])
                y = np.array([header["ylo"], header["yhi"]])
                grid = [x, y, grid]

                contour_lvls = np.arange(0, self.res_range[1] + interval, interval)
                cnt.append(
                    export_contours(
                        grid,
                        z_index=2,
                        levels=contour_lvls,
                        mask=True,
                        fp_mask=self.fps["2d"],
                        layer_mask=f"mask_man_pl",
                        crs=self.crs,
                    )
                )
                # except:
                # print("Error exporting contours..")

            cnt = pd.concat(cnt).reset_index(drop=True)
            layer = f"contours_{type}_ls"
            cnt.to_file(self.fps["2d"], layer=layer)
            print(f"Exported - {self.fps['2d']} - {layer}")

        return self

    def Gpkg_atlas(self, atlas_input):

        ld = get_ld(self.fps["apy"], ext="apd")
        ld["line"] = ld["f"].str.slice(0, 3).astype(int)
        ld = ld.drop_duplicates(subset="line", keep="first")

        xrange = []
        zmax = []
        el_space = []
        line = []

        for fp in ld["fp"]:
            apy = Apy(fp).Load()
            electrodes = apy.electrodes

            xrange.append(electrodes["ld_hor"].max() - electrodes["ld_hor"].min())
            zmax.append(apy.data["z0"].max())
            el_space.append(apy.el_space)
            line.append(apy.line)

        atlas_input["ID_line"] = line
        atlas_input["xrange"] = xrange
        atlas_input["zmax"] = zmax
        atlas_input["el_space"] = el_space

        atlas = pd.DataFrame(atlas_input).sort_index(axis=1)
        atlas["xmin"] = 10000 * atlas["ID_line"]
        atlas["xmax"] = atlas["xmin"] + xrange
        atlas["zmax"] = 5 * np.round(atlas["zmax"] / 5) + atlas["z_gap"]
        # atlas["zmax"] += atlas["z_gap"]
        atlas["map_height"] = (atlas["page_heights"] - 30) / 3

        atlas["zmin"] = atlas["zmax"] - atlas["map_height"] * atlas["map_scales"] / 1000
        atlas["zrange"] = atlas["zmax"] - atlas["zmin"]

        atlas["page_width"] = np.round(
            atlas["xrange"] * (1000 / atlas["map_scales"]) + 25 + 10
        ).astype(int)

        atlas["map_height_int"] = (atlas["page_heights_int"] - 35) / 2

        atlas["qcl_zrange"] = atlas["zrange"].min()
        atlas["qcl_map_height"] = (atlas["page_heights"].min() - 30) / 3

        geom = []
        crds = zip(atlas["xmin"], atlas["xmax"], atlas["zmin"], atlas["zmax"])
        for x1, x2, y1, y2 in crds:
            geom.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

        atlas = gpd.GeoDataFrame(atlas, geometry=geom, crs=self.crs)
        atlas.to_file(self.fps["2d"], engine="pyogrio", layer="atlas_pl")
        print(f"\nExported - {self.fps['2d']} - atlas_pl ")

        self.atlas = atlas
        return self

    # def Merge_contours(self, type):
    #     ld = get_ld("tmp/")
    #     ld = ld.loc[ld["fn"].str.startswith("tmp_cnt")]
    #     print(ld)
    #     gdf = []
    #     for fp in ld["fp"]:
    #         gdf.append(gpd.read_file(fp))

    #     gdf = pd.concat(gdf)
    #     gdf = gdf.loc[gdf["ELEV"] % 50 == 0]
    #     gdf.crs = 8353
    #     gdf.to_file(self.fps["ert"], layer=f"cnt_{type}_full_ls")
    #     return self

    def QC_lines(self):

        for apy in self.apy.values():
            line = int(apy.line)
            ad = self.atlas.copy()
            ad.index = ad["ID_line"].values
            ad = ad.loc[line, :]
            df = apy.data.copy()
            levels = get_levels(df, ad["lvl_frequency"])
            self.fig_qc_lines = fig_qc_lines(self, levels, ad, line)

        return self

    def Export_gpkg_inv(self):

        ld = get_ld(self.fps["grd"], ext=".csv")
        ld["type"] = ld["f"].str.slice(4, None)
        ld["ID_line"] = ld["f"].str.slice(None, 3).astype(int)

        for type in ld["type"].unique():
            ld_type = ld.copy()
            ld_type = ld_type.loc[ld_type["type"] == type]

            gdf = []
            for fp, line in zip(ld_type["fp"], ld_type["ID_line"]):
                df = pd.read_csv(fp)
                df["ID_line"] = line
                gdf.append(df)

            gdf = pd.concat(gdf).reset_index(drop=True)
            gdf.columns = ["ld_hor", "z", "res", "res_log", "ID_line"]
            # gdf["ld_hor"] += gdf["ID_line"] * 10000
            geom = gpd.points_from_xy(gdf["ld_hor"], gdf["z"])
            gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=self.crs)

            gdf["ref_res"] = get_ref_res(gdf["res"], self.res_range)
            gdf = self.Merge_el_space(gdf)

            gdf.to_file(self.fps["ert"], layer=f"inv_{type}_pt", engine="pyogrio")

        return self

    def Get_grid_dictionary(self):

        ld = get_ld(self.fps["grd"], ext=".csv")
        ld["type"] = ld["f"].str.slice(4, None)
        ld["ID_line"] = ld["f"].str.slice(None, 3).astype("Int64")

        keys_type = ld["type"].unique()
        keys_line = ld["ID_line"].unique()

        grid_dict = {key: None for key in keys_type}
        for key in grid_dict.keys():
            grid_dict[key] = {key: None for key in keys_line}

        return grid_dict, ld

    def Export_surface_notes(self):
        atlas = self.atlas_qc

        geom = []
        for x0, x100, y100 in zip(atlas["xmin"], atlas["xmax"], atlas["zmax"]):
            y0 = y100 - 5
            geom.append(shapely.box(x0, y0, x100, y100))

        gdf = gpd.GeoDataFrame(geometry=geom, crs=self.crs)
        gdf["lab"] = None

        gdf.to_file(self.fps["ert"], layer="notes_surface_pl")

        layers = gpd.list_layers(self.fps["ert"])
        layers = layers["name"].values

        if not "notes_surface_man_pl" in layers:
            gdf.to_file(self.fps["ert"], layer="notes_surface_man_pl")
        return self

    def Export_model_template(self):
        gdf = gpd.read_file(self.fps["ert"], layer="mask_man_pl")
        gdf = gdf.explode()
        gdf = gdf.to_file(self.fps["ert"], layer="03_model_pl")
        return self
