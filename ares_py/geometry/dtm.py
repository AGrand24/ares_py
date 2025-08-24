import rasterio
import numpy as np


def tif_read(fp):
    with rasterio.open(fp) as tif:
        z = tif.read(1)
        cols, rows = np.meshgrid(np.arange(tif.width), np.arange(tif.height))
        x, y = rasterio.transform.xy(tif.transform, cols, rows)
        x = x.reshape(cols.shape)
        y = y.reshape(cols.shape)

    return [x, y, z]
