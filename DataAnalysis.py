import os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_path = r'\\srvsas9402\Platform_AN\3_Projekt_1\Innovationsprojektet\Data'
    file_name = 'akm_full_stay.parquet'

    # Load data
    df_samples = pd.read_parquet(os.path.join(data_path, file_name))

    # Describe data
    description = df_samples.describe()
    print(description)

    # Do histograms
    df_hosp = df_samples.loc[(df_samples.required_hosp == 1) & (df_samples.hosp_time_remaining < 30)]
    df_hosp.hist(column='hosp_time_remaining', bins=100)
    plt.show()

