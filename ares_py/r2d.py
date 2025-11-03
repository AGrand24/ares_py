import numpy as np
import pandas as pd
import os


def parse_data(df):
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


def dat_file_parse(fp_apd, project_name):
    fp_ape = str(fp_apd).replace(".apd", ".ape")
    df_data = pd.read_csv(fp_apd, sep="\t")
    line = df_data["ID_line"].iloc[0]
    el_space = df_data["el_space"].iloc[0]
    array = df_data["arr"].iloc[0]

    data = parse_data(df_data)

    df_topo = pd.read_csv(fp_ape, sep="\t")
    topo = parse_topo(df_topo)

    dat_lines = [f"{project_name}_{line}", float(el_space) / 2]

    dat_lines.append(get_array_id(array))
    dat_lines.append(len(data))
    dat_lines.extend([0, 0])
    dat_lines.extend(data)
    dat_lines.append(2)
    dat_lines.append(len(topo))
    dat_lines.extend(topo)
    dat_lines.extend([1, 0, 0, 0, 0])
    dat_lines = [str(l) + "\n" for l in dat_lines]

    return dat_lines


def dat_file_export(dat_lines, fp):

    with open(fp, "w") as file:
        file.writelines(dat_lines)


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
