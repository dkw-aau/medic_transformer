import pandas as pd
import os


def prepare_parquet(data_path, filename):
    print(f'Preparing file {filename}')
    df_luna_tests = pd.read_sas(os.path.join(data_path, f'{filename}.sas7bdat'), encoding='latin-1')
    df_luna_tests.to_parquet(os.path.join(data_path, f'{filename}.parquet'), use_deprecated_int96_timestamps=True)
    print(f'File Prepared\n')
