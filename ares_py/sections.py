import numpy as np


def get_sections(ert, section_electrodes=8):
    electrodes = ert.data.iloc[:, :4]

    x = np.unique(electrodes)
    x = np.sort(x)

    xn = x / ert.el_space

    sec = xn // section_electrodes

    sec = np.column_stack([xn, sec, sec])
    sec[:, 2] = sec[:, 0] - sec[:, 1] * 8

    return sec


def ld2sec(ld, el_space, cable_el=8):
    el = ld // el_space
    sec = el // cable_el
    el_sec = el - (sec * cable_el)

    return np.column_stack([el, sec, el_sec])
