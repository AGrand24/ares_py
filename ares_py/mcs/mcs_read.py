import pandas as pd
import numpy as np
import json

pd.set_option("future.no_silent_downcasting", True)


def mcs_df_concat_template(df):
    with open("ares_py/mcs/mcs_cols.json", "r") as file:
        cols = json.load(file)["ares2_10ch"]

    df_template = pd.DataFrame(columns=cols)
    df = pd.concat([df_template.astype(df.dtypes), df], axis=0)

    return df


def mcs_format(df):
    df = df.astype(str)

    df[["c01", "c02"]] += "**"

    df[["c01", "c01_s"]] = df["c01"].str.split("*", expand=True, n=1)
    df[["c02", "c02_s"]] = df["c02"].str.split("*", expand=True, n=1)

    df["c01_s"] = df["c01_s"].str.replace("*", "")
    df["c02_s"] = df["c02_s"].str.replace("*", "")

    df["geom_factor"] = df["geom_factor"].str.replace(";geom.fact.(dist. 2m):", "")
    df["geom_factor"] = df["geom_factor"].str.replace("-", "")
    df["geom_factor"] = df["geom_factor"].str.split(",")

    df = df.replace(["-", "", "nan"], np.nan)
    df = df.replace("inf", np.inf)
    df = df.replace("-inf", -np.inf)
    df.iloc[:, :13] = df.iloc[:, :13].astype(float) 

    return df


def mcs_get_col_data(header_version, channels):
    with open("ares_py/mcs/mcs_cols.json", "r") as file:
        cols = json.load(file)
        cols = cols[f"{header_version}_{channels}ch"]
    return cols


def mcs_read_data(fp, header_data, header_version):

    data = {}
    for hd in header_data[:, :]:
        skip = hd[0]
        rows = hd[1] - hd[0]
        channels = hd[2]
        cols = mcs_get_col_data(header_version, channels)
        df = pd.read_csv(
            fp, skiprows=skip, nrows=rows, sep="\t", header=None, names=cols
        )
        df = mcs_df_concat_template(df)
        data[channels] = mcs_format(df)
    return data
