from typing import overload, Dict

import pandas as pd


class DataFrameSaver:
    def __init__(self, **kwargs):
        self.args = kwargs

    @overload
    def __call__(self, df: pd.DataFrame, fp: str):
        pass


class DataFrameToCsv(DataFrameSaver):
    def __call__(self, df, fp):
        df.to_csv(fp, **self.args)


class DataFrameToPickle(DataFrameSaver):
    def __call__(self, df, fp):
        df.to_pickle(fp, **self.args)


class DataFrameToExcel(DataFrameSaver):
    def __call__(self, df, fp):
        df.to_excel(fp, )


class DataFrameToLatex(DataFrameSaver):
    def __call__(self, df, fp):
        df.to_latex(fp, index=False)


default_savers = {
    'csv': DataFrameToCsv(index=False),
    'pkl': DataFrameToPickle(),
    'xlsx': DataFrameToExcel(index=False),
    'tex': DataFrameToLatex(index=False),
}

def save_df(
        df: pd.DataFrame,
        fp: str,
        savers: Dict[str, DataFrameSaver] = default_savers,
        fmt: str = None):
    if fmt is None:
        fmt = fp.split('.')[-1]
    assert fmt in savers
    savers[fmt](df, fp)
