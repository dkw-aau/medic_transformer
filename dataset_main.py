from datasetmodule.prepare_corpus import prepare_patients
from datasetmodule.prepare_parquets import prepare_parquet
from config import Config


# Primary pipeline for processing EHR data into a corpus of sequences
if __name__ == '__main__':
    config_file = ['config.ini']
    args = Config(
        file_path=config_file
    )

    # prepare parquet files
    if args.prepare_parquet:
        for file in ['forloeb', 'diagnosis', 'labka', 'ccs', 'luna']:
            prepare_parquet(args.path['data_fold'], file)

    # Create Corpus
    prepare_patients(args.max_sequences, args.corpus_name, args.path['data_fold'])
    exit('Done')
