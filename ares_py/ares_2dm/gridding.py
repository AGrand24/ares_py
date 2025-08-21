import numpy as np


def wget_data_grd(data):
    x = data[6]
    y = data[7]
    z = data[2][:, 0]
    grd_data = [x, y, z]
    return grd_data
