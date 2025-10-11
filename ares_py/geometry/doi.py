import pandas as pd
import numpy as np


def save_doi_data(doi_data):
    df = pd.concat([pd.DataFrame(v, index=len(v) * [k]) for k, v in doi_data.items()])
    df = df.reset_index()
    df.columns = ["array", "n", "z_a", "k", "k_inverse"]
    df.to_pickle("ares_py/geometry/doi.pkl")


def load_doi_data(array):
    array = np.array(array)
    doi_data = pd.read_pickle("ares_py/geometry/doi.pkl")
    doi_data = doi_data.loc[doi_data["array"] == array]
    doi_data = np.array(doi_data)[:, 1:].astype(np.float64)
    return doi_data


def calc_doi_poly(array):
    doi_data = load_doi_data(array)
    x = doi_data[:, 0]
    y = doi_data[:, 1]
    poly = np.poly1d(np.polyfit(x, y, 7))
    return poly


def get_doi_2dm(data):

    doi = np.full(data[0].shape[0], np.nan)
    # doi = doi.reshape(-1)

    for array in np.unique(data[1]):
        poly = calc_doi_poly(array)
        idx = np.where(data[1] == array)
        doi[idx] = poly(data[5][idx]).reshape(-1)

    doi *= data[4]
    doi *= -1
    doi = np.round(doi, 1)
    data.append(doi)
    return data
