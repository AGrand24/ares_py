import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import struct
from pykrige import OrdinaryKriging
from shapely import LineString, MultiPolygon
from skimage import measure


pio.templates.default = "plotly_dark"
pd.options.plotting.backend = "plotly"


def parse_data(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    data = np.column_stack([x, y, z])
    df = pd.DataFrame(data, columns=["x", "y", "z"])

    df = df.dropna()
    df = df.loc[np.abs(df["z"]) != np.inf]
    return df


def arse_grid(data, cell_size):
    x = data["x"]
    y = data["y"]

    x0 = min(x) - 5
    y0 = min(y) - 5
    x100 = max(x) + 5
    y100 = max(y) + 5

    grid_x = np.arange(x0, x100 + cell_size, cell_size)
    grid_y = np.arange(y0, y100 + cell_size, cell_size)

    grid = [grid_x, grid_y]
    return grid


def plt_data_points(data):
    data = data.values
    marker = dict(size=6, color=data[:, 2], colorscale="Turbo", cmin=0, cmax=4)
    plt = go.Scatter(
        x=data[:, 0],
        y=data[:, 1],
        mode="markers",
        marker=marker,
        hovertext=data[:, 2],
        name="Data points",
        showlegend=True,
    )
    return plt


def custom_linear_variogram(params, dist):

    slope = params[0]
    nugget = params[1]
    return slope * dist + nugget


def run_krg(x, y, z, cell_size, slope=1, nugget=0.1, exact=True):
    data = parse_data(x, y, z)
    grid = arse_grid(data, cell_size)

    custom_params = [slope, nugget]

    ok = OrdinaryKriging(
        data["x"],
        data["y"],
        data["z"],
        variogram_model="custom",  # Set to 'custom'
        # variogram_model="linear",  # Set to 'custom'
        verbose=False,
        variogram_function=custom_linear_variogram,
        variogram_parameters=custom_params,
        exact_values=exact,
    )

    txt = ["Kriging data- " + f"cs={cell_size}"]
    txt.append(f"x=[{grid[0].min()},{grid[0].max()}]")
    txt.append(f"y=[{grid[1].min()},{grid[1].max()}]")
    txt.append(f"cells={grid[0].shape[0]}x{grid[1].shape[0]}")
    txt.append(f"{slope}x+{nugget}")
    txt = (", ").join(txt)
    print(txt)

    grid.extend(ok.execute("grid", grid[0], grid[1]))

    return grid


def export_surfer(grid, z_index, fp):
    """
    Saves a NumPy 2D array to a Surfer 6 Binary Grid file.

    Args:
        data_array (np.ndarray): The 2D NumPy array of grid values.
                                 The shape should be (ny, nx).
        x_coords (np.ndarray): 1D array of the X-coordinates for the columns.
        y_coords (np.ndarray): 1D array of the Y-coordinates for the rows.
        output_file (str): Path to the output .grd file.
    """
    data_array = grid[z_index]

    # Surfer's NoData value
    no_data_value = 1.70141e38

    # Get grid dimensions from the array shape
    ny, nx = data_array.shape

    # Get spatial extents from coordinate arrays
    xlo, xhi = grid[0].min(), grid[0].max()
    ylo, yhi = grid[1].min(), grid[1].max()

    # Handle NaN values and find Z extents
    # Replace any np.nan with Surfer's NoData value
    grid_data = np.nan_to_num(data_array, nan=no_data_value)

    # Find min/max Z values, ignoring the NoData value
    zlo = grid_data[grid_data != no_data_value].min()
    zhi = grid_data[grid_data != no_data_value].max()

    # Open the file for binary writing
    with open(fp, "wb") as f:
        # --- Write the Header ---
        # Pack the header values into a binary string.
        # '<' denotes little-endian byte order.
        # '4s' = 4-char string
        # 'h' = short integer (2 bytes)
        # 'd' = double (8 bytes)
        header_format = "<4shh6d"
        header = struct.pack(
            header_format,
            b"DSBB",  # Binary file identifier
            nx,
            ny,
            xlo,
            xhi,
            ylo,
            yhi,
            zlo,
            zhi,
        )
        f.write(header)

        # --- Write the Data ---
        # Ensure data is in 4-byte float format and flatten it.
        # Surfer grids are written from bottom-left, row by row.
        # NumPy arrays are indexed from top-left, so we don't need to flip.
        flat_data = grid_data.astype(np.float32).flatten()

        # Write the flattened array to the file
        f.write(flat_data.tobytes())


def plot(data, grid):
    fig = go.Figure()
    plt_pts = plt_data_points(data)
    fig = fig.add_trace(plt_pts)
    fig = fig.add_trace(
        go.Heatmap(
            z=grid[2],
            x=grid[0],
            y=grid[1],
            colorscale="Turbo",
            zmin=0,
            zmax=4,
            showlegend=True,
        )
    )
    fig = fig.update_yaxes(scaleanchor="x1", scaleratio=1)
    fig = fig.update_layout(
        height=900,
        width=1600,
    )
    fig.write_html("grid_compare.html")


def export_contours(
    grid,
    levels,
    z_index=2,
    mask=False,
    fp_mask=None,
    layer_mask=None,
    crs=8353,
):

    grid_z = np.transpose(grid[z_index])
    x0 = grid[0].min()
    y0 = grid[1].min()

    geom = []
    attributes = []
    for lvl in levels:
        contours = measure.find_contours(grid_z, lvl)

        contours = [
            cnt
            + np.column_stack([np.full(cnt.shape[0], x0), np.full(cnt.shape[0], y0)])
            for cnt in contours
        ]

        geom.extend([LineString(contour) for contour in contours])
        attributes.extend([{"level": lvl, "id": i} for i, _ in enumerate(contours)])

    gdf = gpd.GeoDataFrame(attributes, geometry=geom, crs=crs)

    if mask == True:
        gdf_mask = gpd.read_file(fp_mask, layer=layer_mask)
        geom = [MultiPolygon(gdf_mask["geometry"])]
        gdf_mask = gpd.GeoDataFrame(geometry=geom, crs=crs)
        gdf = gpd.clip(gdf, gdf_mask)

    return gdf


def read_surfer6_binary_grid(filename):
    """
    Reads a Surfer 6 binary grid file and returns the grid data as a NumPy array.

    Args:
        filename (str): The path to the Surfer 6 binary grid file.

    Returns:
        numpy.ndarray: A 2D NumPy array containing the grid data.
        dict: A dictionary containing the header information.
    """
    with open(filename, "rb") as f:
        # Read the header
        header_format = "<4s2h6d"
        header_size = struct.calcsize(header_format)
        header_bytes = f.read(header_size)

        if not header_bytes:
            raise ValueError("File is empty or header is incomplete.")

        header_data = struct.unpack(header_format, header_bytes)

        magic_word = header_data[0].decode("ascii")
        if magic_word != "DSBB":
            raise ValueError("Not a valid Surfer 6 binary grid file.")

        nx = header_data[1]
        ny = header_data[2]
        xlo = header_data[3]
        xhi = header_data[4]
        ylo = header_data[5]
        yhi = header_data[6]
        zlo = header_data[7]
        zhi = header_data[8]

        header_info = {
            "nx": nx,
            "ny": ny,
            "xlo": xlo,
            "xhi": xhi,
            "ylo": ylo,
            "yhi": yhi,
            "zlo": zlo,
            "zhi": zhi,
        }

        # Read the grid data
        data = np.fromfile(f, dtype=np.float32)

        # Reshape the data into a 2D array
        grid_data = data.reshape((ny, nx))

        return grid_data, header_info
