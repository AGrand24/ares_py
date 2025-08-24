import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd

pio.templates.default = "plotly_dark"


def plt_meas(ert, clr="res"):
    df = ert.data.copy()
    # if not clr in ["res"]:
    #     df = df.loc[df["res"] > 0]
    x = df["ld_hor"]
    y = df["z"]

    clr_def = get_clr_def(df, clr, ert.cs_res)
    cd = get_cd(df)
    ht = get_ht(clr_def)

    marker = dict(
        color=clr_def["data"],
        colorscale=clr_def["cs"],
        size=10,
        symbol="square",
        cmin=clr_def["cmin"],
        cmax=clr_def["cmax"],
        showscale=clr_def["showscale"],
        colorbar=clr_def["colorbar"],
    )
    plt = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        customdata=cd,
        hovertemplate=ht,
        name=clr,
        marker=marker,
        visible=clr_def["visible"],
        showlegend=True,
    )
    return plt


def plt_meas_3d(df, cs_res, clr="res"):

    x = df["x"]
    y = df["y"]
    z = df["z"]

    clr_def = get_clr_def(df, clr, cs_res)
    cd = get_cd(df)
    ht = get_ht(clr_def, type="3d")

    marker = dict(
        color=clr_def["data"],
        colorscale=clr_def["cs"],
        size=5,
        symbol="square",
        cmin=clr_def["cmin"],
        cmax=clr_def["cmax"],
        showscale=clr_def["showscale"],
        colorbar=clr_def["colorbar"],
    )
    plt = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        customdata=cd,
        hovertemplate=ht,
        name=clr,
        marker=marker,
        visible=clr_def["visible"],
        showlegend=True,
    )
    return plt


def plt_electrodes(ert, topo=False):

    xn = ert.sec[:, 0] + 1
    sec = ert.sec[:, 1] + 1
    n_sec = ert.sec[:, 2] + 1

    x = ert.sec[:, 0] * ert.el_space
    y = np.tile(4, xn.shape[0])

    cd = np.column_stack([x, y - 4, xn, sec, n_sec])
    ht = "x: %{cd[0]}, z: %{cd[1]}<br>n: %{cd[2]}, sec: %{cd[3]}<br>n sec: %{cd[4]}"
    ht = ht.replace("cd", "customdata")

    plt = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=7,
            color=sec,
            colorscale="Rainbow",
            line=dict(width=1, color="#6B6B6B"),
        ),
        name="electrodes",
        customdata=cd,
        hovertemplate=ht,
    )

    return plt


def plt_dtm(dtm):

    x = dtm[0]
    y = dtm[1]
    z = dtm[2]
    plt = go.Heatmap(
        z=z,
        x0=np.min(x),
        y0=np.min(y),
        dx=np.ptp(x) / z.shape[1],
        dy=np.ptp(y) / z.shape[0],
    )
    return plt


def plt_topo(ert, visible=True):
    y1 = ert.sec["topo"]
    y2 = ert.sec["dtm"]
    y3 = ert.sec["dtm_dist"]
    x = ert.sec["ld_hor"]

    marker = dict(size=4)

    plt = [
        go.Scatter(
            x=x, y=y1, mode="markers", marker=marker, name="topo", visible=visible
        )
    ]
    if ert.check["dtm"] == True:
        plt.append(
            go.Scatter(
                x=x, y=y2, mode="markers", marker=marker, name="dtm", visible=visible
            )
        )
        plt.append(go.Scatter(x=x, y=y3, name="sample<br>dist.", visible="legendonly"))
        plt.append(go.Scatter(x=x, y=y2 - y1, name="z diff", visible="legendonly"))
    return plt


def get_clr_def(df, clr, cs_res):

    clr_def = {
        "res": {
            "data": df["res"],
            "cd": "%{cd[5]}",
            "cs": cs_res["color_scale"],
            "cmin": cs_res["crange"][0],
            "cmax": cs_res["crange"][1],
            "showscale": False,
            "colorbar": dict(
                tickvals=cs_res["tickvals"],
                ticktext=cs_res["ticktxt"],
            ),
            "visible": True,
        },
        "res_log": {
            "data": np.log10(df["res"].clip(lower=1)),
            "cd": "%{cd[5]}",
            "cs": "Turbo",
            "cmin": None,
            "cmax": None,
            "showscale": False,
            "colorbar": dict(
                tickvals=cs_res["tickvals"],
                ticktext=cs_res["ticktxt"],
            ),
            "visible": "legendonly",
        },
        "ep": {
            "data": df["ep"],
            "cd": "%{cd[8]}",
            "cs": "RdBu_r",
            "cmin": -500,
            "cmax": 500,
            "showscale": False,
            "colorbar": dict(
                tickvals=cs_res["tickvals"],
                ticktext=cs_res["ticktxt"],
            ),
            "visible": "legendonly",
        },
        "std": {
            "data": df["std"],
            "cd": "%{cd[9]}",
            "cs": "Temps",
            "cmin": 0,
            "cmax": 1,
            "showscale": False,
            "colorbar": dict(
                tickvals=cs_res["tickvals"],
                ticktext=cs_res["ticktxt"],
            ),
            "visible": "legendonly",
        },
    }
    return clr_def[clr]


def get_cd(df):
    cd = df.copy().fillna(-1)
    cd["i"] = (1000 / cd["i"]).round()

    round = [("res", 0), ("v", 0), ("ep", 0), ("i", 0), ("a", 0), ("n", 2), ("z", 0)]
    for col, d in round:
        cd[col] = np.round(cd[col], d)
        cd[col] = cd[col].astype(str).str.rstrip(".0")

    cd["std"] = np.round(cd["std"], 1)

    cd = cd.astype(str)
    cd = cd.values
    return cd


def get_ht(clr_def, type="2d"):

    ht = [clr_def["cd"]]
    if type == "2d":
        ht.append("%{x}, %{y}, %{cd[16]} ")
    else:
        ht.append("%{x}, %{y}, %{z}, %{cd[16]} ")

    ht.append("%{cd[0]} %{cd[1]} %{cd[2]} %{cd[3]}")
    ht.append("R: %{cd[5]}, V: %{cd[6]}, I: %{cd[7]}")
    ht.append("SP: %{cd[8]}, Vout: %{cd[10]}")
    ht.append("a: %{cd[13]}, n: %{cd[14]}")
    ht.append("std: %{cd[9]}, ch: %{cd[11]} %{cd[12]}")
    ht.append("ID: %{cd[17]}")
    ht = ("<br>").join(ht)
    ht = ht.replace("cd[", "customdata[")
    return ht


def plt_dtm_3d(df_dtm):
    plt = go.Scatter3d(
        x=df_dtm[0],
        y=df_dtm[1],
        z=df_dtm[2],
        mode="markers",
        marker=dict(color=df_dtm[2], colorscale="Spectral_r", size=5),
        name="dtm",
        visible="legendonly",
    )
    return plt


def plt_sec_3d(sec):

    cd = sec[["ld", "sec", "dtm"]].values
    ht = "%{x}, %{y}, %{customdata[2]}<br>x el.: %{customdata[0]}<br>sec.: %{customdata[1]}<br>"

    plt = go.Scatter3d(
        x=sec["x"],
        y=sec["y"],
        z=sec["dtm"] + 2,
        mode="markers",
        marker=dict(color=sec["sec"], colorscale="Turbo", size=2),
        name="el",
        visible="legendonly",
        customdata=cd,
        hovertemplate=ht,
    )
    return plt
