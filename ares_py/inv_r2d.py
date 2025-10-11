import numpy as np
import pandas as pd
import os


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

    lines = [f"{ert.project.name}_{ert.line}", float(ert.el_space) / 2]
    lines.append(get_array_id(ert.arrays[0]))
    lines.append(ert.data.shape[0])
    lines.extend([0, 0])

    lines.extend(format_data(ert.data))
    lines.append(2)
    lines.append(len(ert.topo))
    lines.extend(parse_topo(ert.topo))
    lines.extend([1, 0, 0, 0, 0])
    lines = [str(l) + "\n" for l in lines]
    return lines


def export_to_res2d(ert):
    lines = parse_dat_lines(ert)

    with open(ert.fps["r2d_dat"], "w") as file:
        file.writelines(lines)


def format_r2d_output(ert):
    fp_in = ert.fps["r2d_out"]
    fp_out = ert.fps["inv_r2d"]
    offset = int(ert.line) * 10000

    if os.path.exists(fp_in):
        df = pd.read_csv
        df = pd.read_fwf(fp_in, skiprows=1, header=None)

        df.columns = ["x", "z", "res"]
        df["x"] += offset

        df.to_csv(fp_out, sep="\t", index=False)
