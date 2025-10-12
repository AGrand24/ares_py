import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import plotly.io as pio
import geopandas as gpd
from pathlib import Path
import numpy as np


def get_levels(df, lvl_frequency):
    df["lab"] = "z= " + np.abs(np.round(df["doi"])).astype(int).astype(str).str.zfill(2)
    cols = ["ld_hor", "res", "lab"]
    levels = df.groupby("doi")[cols].apply(np.array).to_list()
    levels = levels[::lvl_frequency]
    return levels


def fig_qc_lines(prj, levels, ad, id_line):
    rrange = [ad["res_min"], (ad["res_max"] * 1.05)]
    xrange = [ad["xmin"], ad["xmax"]]
    xrange = [x % 10000 for x in xrange]

    colors = sample_colorscale("Turbo_r", np.linspace(0, 1, len(levels)))

    fig = go.Figure()
    for i, lvl in enumerate(levels):
        name = lvl[0][2]
        data = np.vstack(lvl)
        data[:, 1] = np.clip(data[:, 1], min=rrange[0], max=rrange[1])
        marker = dict(color=colors[i], size=10)
        line = dict(width=3)
        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                name=name,
                mode="lines+markers",
                marker=marker,
                line=line,
            )
        )

    fig.update_layout(
        legend=dict(
            x=0.02,
            y=0.98,
            orientation="h",
            font=dict(size=20),
            xanchor="left",
            yanchor="top",
        ),
        # yaxis_range=np.log10(yrange),
        yaxis_range=rrange,
        xaxis_range=xrange,
        xaxis=dict(visible=False),
    )
    # fig.update_yaxes(type="log")

    width = ad["xrange"] * 10
    height = width * ad["zrange"] / ad["xrange"]

    fig.update_layout(
        width=5000,
        height=5000 * ad["zrange"] / ad["xrange"],
        margin=dict(t=0, b=0, r=0, l=0),
        template="plotly_white",
        yaxis=dict(ticklabelposition="inside"),
    )

    fig.update_yaxes(showgrid=True, gridwidth=6, tickfont=dict(size=25), nticks=20)

    fig.update_yaxes(tickprefix="       ")

    fp = Path(prj.fps["qcl"], f"{str(id_line).zfill(3)}.png")
    print(fp)
    fig.write_image(fp, width=width, height=height)

    lines = [(ad["xrange"] + 2) / width]
    lines.append(0.0)
    lines.append(0.0)
    lines.append(-(ad["zrange"]) / height)
    lines.append(ad["xmin"] - 2)
    lines.append(ad["zmax"])

    lines = [str(l) + "\n" for l in lines]

    fp = Path(prj.fps["qcl"], f"{str(id_line).zfill(3)}.pgw")

    with open(fp, "w") as file:
        file.writelines(lines)
    return fig
