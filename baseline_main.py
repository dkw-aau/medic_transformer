from baselinemodule.Baseline import Baseline
from config import Config

# Primary pipeline for processing EHR data into a corpus of sequences
if __name__ == '__main__':
    config_file = ['config.ini']
    args = Config(
        file_path=config_file
    )

    # Train Baseline
    baseline = Baseline(args)
    baseline.train()