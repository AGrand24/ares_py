import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

pd.set_option("future.no_silent_downcasting", True)

# pio.templates.default = "plotly_dark"

cs = [
    [0, "#FFFFFF"],
    [0.01, "#0400E4"],
    [100, "#0400E4"],
    [250, "#00FFFF"],
    [500, "#00ff7b"],
    [1000, "#f2ff00"],
    [5000, "#ff0000"],
    [10000, "#fb00b0"],
]


def generate_clr_scale(cs, fig_show=False, save=False):
    cs = pd.DataFrame(np.vstack(cs))
    cs[0] = cs[0].astype(float)
    cmax = cs[0].max()
    cs[2] = cs[0] / cs[0].max()
    ticks = np.full(cs.shape[0], 0)
    ticks[0, -1] = 1

    if save == True:
        name = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        fp = f"ares_py/colorscales/cs_{name}.tsv"
        cs.to_csv(fp, sep="\t", index=False, header=None)

    cs = [tuple((x, y)) for x, y in zip(cs[2], cs[1])]

    x = np.arange(0, cmax + 25, 25)
    plt = go.Scatter(x=x, y=x, mode="markers", marker={"color": x, "colorscale": cs})
    fig = go.Figure(plt)

    if fig_show == True:
        fig.show()
    return cs


def load_clr_scale(cs):
    name = cs["name"]
    fp = f"ares_py/colorscales/{name}.tsv"

    df = pd.read_csv(fp, sep="\t", header=None)
    cmin = df[0].min()
    cmax = df[0].max()
    cs = [tuple((x, y)) for x, y in zip(df[2], df[1])]

    tickvals = df.loc[df.iloc[:, 3] == 1, 0].to_list()
    ticktxt = [str(int(t)) for t in tickvals]

    cs = dict(
        name=name,
        color_scale=cs,
        crange=[cmin, cmax],
        tickvals=tickvals,
        ticktxt=ticktxt,
    )

    return cs
