import os
import pandas as pd
from pathlib import Path


def get_ld(dir_path, **kwargs):
    df = pd.DataFrame(columns=["fp", "fn", "f", "ext", "fp_abs"])
    file_path = []
    for root, dir, files in os.walk(dir_path):
        for file in files:
            file_path.append(os.path.join(root, file))

    fn = []
    f = []
    ext = []
    fp_abs = []

    for fp in file_path:
        fn.append(Path(fp).name)
        f.append(Path(fp).stem)
        ext.append(Path(fp).suffix)
        fp_abs.append(Path(fp).absolute())

    data = {"fp": file_path, "fn": fn, "f": f, "ext": ext, "fp_abs": fp_abs}
    df = pd.concat([df, pd.DataFrame(data)], axis=0)
    df = df.astype(str)

    if kwargs.get("ext", None) != None:
        df = df.loc[df["fp"].str.endswith(kwargs.get("ext", None))]

    return df.reset_index(drop=True)
