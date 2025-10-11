import numpy as np
import geopandas as gpd
from shapely import MultiPoint, concave_hull
from pykrige import OrdinaryKriging
import struct
import shapely
import plotly.graph_objects as go

# version 20250822


def ag_kriging(data, cell_size=1, variogram_model="exponential", exact=True):
    x = data[0]
    y = data[1]
    z = data[2]

    x0 = min(x) - 5
    y0 = min(y) - 5
    x100 = max(x) + 5
    y100 = max(y) + 5

    cells_x = int((x100 - x0) / cell_size)
    cells_y = int((y100 - y0) / cell_size)

    grid_x = np.linspace(x0, x100, cells_x)
    grid_y = np.linspace(y0, y100, cells_y)

    print("Gridding data:")
    print(f"\tcell_size={cell_size}")
    print(f"\tx={x0},{x100}")
    print(f"\ty={y0},{y100}")
    print(f"\tcells={grid_x.shape},{grid_y.shape}")

    ok = OrdinaryKriging(x, y, z, variogram_model=variogram_model, exact_values=exact)

    grid_z, ss = ok.execute("grid", grid_x, grid_y)
    mesh = np.meshgrid(grid_x, grid_y)

    # mesh[2] = grid_z.data
    return [mesh[0], mesh[1], grid_z.data]


def ag_mask(grid, poly_mask):

    grid_x = grid[0]
    grid_y = grid[1]
    grid_z = grid[2]

    z_flat = grid_z.flatten()

    points = gpd.GeoSeries(gpd.points_from_xy(grid_x.flatten(), grid_y.flatten()))
    mask = poly_mask.contains(points)
    mask = np.where(mask == False)
    z_flat[mask] = np.nan
    grid_z_masked = z_flat.reshape(grid_z.shape)
    grid.append(grid_z_masked)
    return grid


def ag_pt2pl(coords: list, ratio=0.2, buffer=1):
    geometry = gpd.points_from_xy(coords[0], coords[1])
    pl = concave_hull(MultiPoint(geometry), ratio).buffer(buffer)
    return pl


def ag_export_surfer(grid, z_index, fp):
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


def ag_plot(grd, poly_mask, w=500, h=500, crange=[None, None]):
    poly_mask_coords = shapely.get_coordinates(poly_mask)

    fig = go.Figure()
    plt = []

    for i in list(range(2, len(grd))):
        plt.append(
            go.Heatmap(
                z=grd[i],
                dx=np.ptp(grd[0]) / grd[0].shape[1],
                dy=np.ptp(grd[1]) / grd[0].shape[0],
                x0=grd[0].min(),
                y0=grd[1].min(),
                colorscale="Spectral_r",
                zmin=crange[0],
                zmax=crange[1],
                showlegend=True,
            )
        )

    plt.append(
        go.Scatter(x=poly_mask_coords[:, 0], y=poly_mask_coords[:, 1], hoverinfo="skip")
    )
    fig.add_traces(plt)

    fig.update_yaxes(scaleanchor="x1", scaleratio=1)
    fig.update_layout(width=w, height=h)
    fig.show()

    return fig
