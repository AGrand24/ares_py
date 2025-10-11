import pandas as pd
import numpy as np

from ares_py.geometry.doi import get_doi_2dm
from ares_py.geometry.geometry_2dm import get_n_a_2dm, get_x_meas


def read_ascii(fp):
    with open(fp) as file:
        lines = file.readlines()
        lines = [l.strip() for l in lines]
        lines = pd.Series(lines)

        idx_header = lines.loc[lines.str.startswith("C1")].index[0]

        header = lines[:idx_header]
        data = lines[idx_header + 1 :]
        data = pd.DataFrame(data).iloc[:, 0].str.split("\t", expand=True)
        data = np.array(data)
    return data, header


def version_2dm(header):
    version = "a1"
    if "ARES II" in header[0]:
        version = "a2"
    return version


# Res,U,I,Ep,Std,U_out


def split_2dm_data(data, version):
    if version == "a1":
        idx = 4
        electrodes = data[:, :idx]
        array = data[:, idx]

        meas = data[:, idx + 1 :]
        meas = meas[:, [3, 1, 0, 2, 4]]
        # add empty col to match a2 shape
        meas = np.column_stack([meas, np.full((meas.shape[0], 1), fill_value=np.nan)])

        channel = np.tile(["P1", "P2"], electrodes.shape[0])
        channel = channel.reshape(-1, 2)
    if version == "a2":
        idx = 4
        electrodes = data[:, :idx]
        electrodes = format_electrodes_a2(electrodes)

        array = data[:, idx + 2]
        channel = data[:, idx : idx + 2]
        meas = data[:, idx + 3 : idx + 9]
        meas = meas[:, [4, 2, 1, 3, 5, 0]]

    electrodes = electrodes.astype(np.int64)
    array = array.astype(str)
    array = np.char.lower(array)
    meas = meas.astype(np.float32)
    channel = channel.astype(str)

    return [electrodes, array, meas, channel]


def format_electrodes_a2(electrodes):
    c1 = pd.DataFrame(electrodes[:, 0]).iloc[:, 0].str.split("*", expand=True)
    c2 = pd.DataFrame(electrodes[:, 1]).iloc[:, 0].str.split("*", expand=True)

    c1 = np.array(c1)[:, 0]
    c2 = np.array(c2)[:, 0]

    electrodes[:, [0, 1]] = np.column_stack([c1, c2])
    electrodes = electrodes.astype(np.int64)
    return electrodes


def format_header(header):
    h = header.str.split(":", expand=True, n=1)
    rename = rename = [
        ("Distance", "electrode_distance"),
        ("Length", "length_pf"),
        ("MC-set", "method"),
        ("Profile length", "length_pf"),
        ("Pulse length", "pulse"),
    ]
    for old, new in rename:
        h.iloc[:, 0] = h.iloc[:, 0].replace(old, new)

    h[0] = h[0].str.replace(" ", "_")
    h[0] = h[0].str.replace("-", "_")
    h[0] = h[0].str.lower()
    header = pd.Series(h[1], name="value")
    header.index = h[0]
    header = header.str.strip()
    return header


def read_2dm(fp):
    data, header = read_ascii(fp)
    version = version_2dm(header)
    data = split_2dm_data(data, version)
    header = format_header(header)

    spacing = get_el_spacing(header)
    data[0] = data[0] * spacing
    data = get_n_a_2dm(data)
    data = get_x_meas(data)
    data = get_doi_2dm(data)
    data[2][:, 0] = np.round(data[2][:, 0])

    return data, version, header, spacing


def get_el_spacing(header):
    spacing = header["electrode_distance"]
    spacing = spacing.replace("m", "").strip()
    spacing = int(spacing)
    return spacing


def delete_empty_entires(ert):
    mask1 = ert.data["res"] != 0
    mask2 = ert.data["i"] != 0
    mask3 = ert.data["v"] != 0
    mask4 = ert.data["ep"] != 0
    return ert.data.loc[mask1 & mask2 & mask3 & mask4]


def load_2dm(ert):
    read = read_2dm(ert.fp_load)
    data = read[0]
    ert.version = read[1]
    ert.header = read[2]
    ert.el_space = read[3]

    columns = [
        "c1",
        "c2",
        "p1",
        "p2",
        "arr",
        "res",
        "v",
        "i",
        "ep",
        "std",
        "v_out",
        "ch1",
        "ch2",
        "a",
        "n",
        "ld",
        "doi",
    ]
    ert.data = pd.concat([pd.DataFrame(d) for d in data], axis=1)
    ert.data.columns = columns

    ert.data = delete_empty_entires(ert)

    ert.data.iloc[:, :4] = ert.data.iloc[:, :4].astype("Int64")
    ert.data["ID_meas"] = list(range(ert.data.shape[0]))
    ert.data["ID_meas"] = ert.data["ID_meas"].astype(str).str.zfill(5)
    ert.data["ID_meas"] = ert.line + "_" + ert.data["ID_meas"]

    ert.data["z"] = ert.data["doi"]
    ert.data["ld_hor"] = ert.data["ld"]
    ert.data["ID_line"] = int(ert.line)
    ert.arrays = np.unique(ert.data["arr"])
    return ert
