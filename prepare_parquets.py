import pandas as pd
import os


def prepare_parquet(filename):
    print(f'Preparing file {filename}')
    df_luna_tests = pd.read_sas(os.path.join(data_path, f'{filename}.sas7bdat'), encoding='latin-1')
    df_luna_tests.to_parquet(os.path.join(data_path, f'{filename}.parquet'), use_deprecated_int96_timestamps=True)
    print(f'File Prepared\n')


if __name__ == '__main__':
    # File paths
    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'

    # Prepare parquet from sas files
    #prepare_parquet('forloeb')
    #prepare_parquet('diagnosis')
    #prepare_parquet('luna')
    #prepare_parquet('labka')
    prepare_parquet('ccs')
