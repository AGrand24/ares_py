import geopandas as gpd
from ares_py.tools.geometry_tools import pt_to_ls


def export_topo_ls(prj):

    df = prj.sec[["ID_line", "ld_hor", "z0", "ld"]].copy()
    df["ld_hor"] += df["ID_line"].astype(float) * 10000

    gdf_ls = pt_to_ls(df, x="ld_hor", y="z0", order="ld", groupby="ID_line", crs=8353)
    gdf_ls.to_file(prj.fp["ert"], layer="ert2d_topo_ls", engine="pyogrio")


def export_topo_pt(prj):

    df = prj.sec[["ID_line", "ld_hor", "z0", "ld", "n_el"]].copy()
    df["ld_hor"] += df["ID_line"].astype(float) * 10000

    geom = gpd.points_from_xy(x=df["ld_hor"], y=df["z0"])
    gdf_pt = gpd.GeoDataFrame(df, geometry=geom, crs=8353)
    gdf_pt.to_file(prj.fp["ert"], layer="ert2d_topo_pt", engine="pyogrio")
