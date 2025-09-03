from shapely import LineString
import geopandas as gpd


def pt_to_ls(df, x, y, order, groupby, cols=None, crs=3857, z=None):
    if z == None:
        df["z"] = 0
    else:
        df["z"] = df[z]

    df = df.sort_values(order)
    gb = df.groupby(groupby)[[x, y, "z"]].agg(list)

    id_line = gb.index

    geom = []
    for x, y, z in zip(gb[x], gb[y], gb["z"]):
        geom.append(LineString(zip(x, y, z)))

    ls = gpd.GeoDataFrame(data=id_line, geometry=geom, crs=crs)

    return ls
