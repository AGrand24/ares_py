import numpy as np


def get_sections(ert, section_electrodes=8):
    electrodes = ert.el

    x = np.unique(electrodes)
    x = np.sort(x)

    xn = x / ert.el_space

    sec = xn // section_electrodes

    sec = np.column_stack([xn, sec, sec])
    sec[:, 2] = sec[:, 0] - sec[:, 1] * 8

    return sec
