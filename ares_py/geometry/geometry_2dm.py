import numpy as np


def get_n_a_2dm(data):
    ind_a = {
        "ws": (3, 2),
        "dd": (0, 1),
        "pp": (2, 1),
        "pd": (3, 2),
        "wa": (1, 0),
        "wb": (1, 0),
        "wc": (1, 0),
    }
    ind_n = {
        "ws": (2, 0),
        "dd": (2, 0),
        "pp": (2, 1),
        "pd": (2, 1),
        "wa": (1, 0),
        "wb": (1, 0),
        "wc": (1, 0),
    }
    a = np.full(data[0].shape[0], np.nan)
    n = np.full(data[0].shape[0], np.nan)

    for array in np.unique(data[1]):
        idx = np.where(data[1] == array)

        a1 = data[0][idx, ind_a[array][0]]
        a2 = data[0][idx, ind_a[array][1]]
        a3 = a1 - a2

        n1 = data[0][idx, ind_n[array][0]]
        n2 = data[0][idx, ind_n[array][1]]
        n3 = n1 - n2

        a[idx] = a3
        n[idx] = n3

    # a = a.reshape(-1)
    # n = n.reshape(-1)
    n = n / a

    data.append(a)
    data.append(n)
    return data


def get_x_meas(data):
    x = np.full((data[0].shape[0]), np.nan)

    for array in np.unique(data[1]):
        idx = np.where(data[1] == array)

        if array in ["pd", "pp"]:
            if array == "pp":
                x1 = data[0][idx, 0]
                x2 = data[0][idx, 2]
            else:
                x1 = data[0][idx, 0]
                x21 = data[0][idx, 2]
                x22 = data[0][idx, 3]
                x2 = (x21 + x22) / 2

        else:
            x1 = np.nanmax(data[0][idx], axis=1)
            x2 = np.nanmin(data[0][idx], axis=1)

        x3 = (x1 + x2) / 2
        x[idx] = x3.reshape(-1)
    x = np.round(x, 2)
    data.append(x)
    return data
