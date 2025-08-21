import numpy as np
import pandas as pd


def get_channels(lines, data_index):

    ch = pd.Series([lines[i] for i in data_index[:, 0] - 2])
    ch = ch.str.replace(r"\D", "", regex=True).astype("Int64")
    ch = np.array(ch)
    return ch


def get_ares2_data_index(channel_header, lines):
    index_start = np.array(channel_header.index.astype(int)) + 1
    index_end = index_start.copy() - 4
    index_end[0] = len(lines)
    index_end = np.roll(index_end, -1)
    index = np.concatenate((index_start, index_end), axis=0)
    index = index.reshape(2, len(channel_header))
    index = np.transpose(index)
    return index


def mcs_get_header_version(lines):
    channel_header = lines.loc[lines.str.contains("C1\tC2\tP1\tP2")]

    if channel_header.empty == True:
        header_version = "ares1"
        data_index = np.array([5, len(lines)]).reshape(1, 2)
        channels = np.array([1])
    else:
        header_version = "ares2"
        data_index = get_ares2_data_index(channel_header, lines)
        channels = get_channels(lines, data_index)
    channels = channels.reshape(channels.shape[0], 1)
    header_data = np.concatenate([data_index, channels], axis=1)
    return header_version, header_data
