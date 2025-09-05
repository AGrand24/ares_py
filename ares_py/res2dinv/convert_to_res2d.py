import numpy as np
import pandas as pd


def format_data(df):
    e0 = np.min(df.iloc[:, :4], axis=1)
    data = np.column_stack([e0, df["a"], df["n"], df["res"]])
    data = pd.DataFrame(data).astype(str)
    data = data.agg(("\t").join, axis=1)
    data = data.to_list()
    return data


def parse_topo(df):
    df = df[["ld", "z0"]].astype(str)
    topo = df.agg(("\t").join, axis=1)
    topo = topo.to_list()
    return topo


def get_array_id(array):
    array_id = {
        "ws": 7,
        "dd": 3,
        "pp": 2,
        "pd": 6,
        "wa": 1,
        "wb": 4,
        "wc": 5,
    }
    return array_id[array]


def parse_dat_lines(ert):

    lines = [f"{ert.project}_{ert.line}", float(ert.el_space)]
    lines.append(get_array_id(ert.arrays[0]))
    lines.append(ert.data.shape[0])
    lines.extend([0, 0])

    lines.extend(format_data(ert.data))
    lines.append(2)
    lines.append(len(ert.sec))
    lines.extend(parse_topo(ert.sec))
    lines.extend([1, 0, 0, 0, 0])
    lines = [str(l) + "\n" for l in lines]
    return lines


def export_to_res2d(ert):
    lines = parse_dat_lines(ert)

    with open(ert.fp["r2d_dat"], "w") as file:
        file.writelines(lines)
