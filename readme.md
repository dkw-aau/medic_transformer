# Medic-BERT

This repository contains the experimental code for the Medic-BERT transformer model
for predicting the length of hospitalization stays (LOS) for patients based on sequences of 
medical events. 

Given the patients' EHR data as medical events, the program is able to train 
predictive models for LOS prediction. 

### Installation
The software is written in python 3.10. 

Dependencies can be installed through the `requirements.txt` file.

### Placing data files
Two csv structured EHR data files should be placed in the `Data` folder.
Currently, CSV and PARQUET formats are supported. 

`patients.csv` specifies the patient cohort and should contain the following fields:

| sequence_id | patient_id | age | sex | hosp_start | los |
|-------------|------------|-----|-----|------------|-----|
* `sequence_id (int)`- The id of the sequence
* `patient_id (int)`- The id of the patient
* `age (int)`- The age of the patient in years
* `sex (0/1)`- Binarized patient sex
* `hosp_start (datetime)`- Start date and time of the hospitalization
* `los (float)`- Length of stay in days

`data.csv` specifies the EHR of a patient modelled as individual tokenized medical events

| token | token_orig | event_time | event_type | event_value | sequence_id |
|-------|------------|------------|------------|-------------|-------------|
* `token (str)`- The tokenized version of a medical event (e.g., Temp_Low)
* `token_orig (str)`- The original token before tokenization (e.g., Temp) 
* `event_time (datetime)`- Date and time of the event
* `event_type (str)`- The type of the event (e.g., Laboratory)
* `event_value (float)`- Original float/categorical value of event
* `sequence_id (int)`- The id of the sequence

Example data files can be located in the `Data` folder.

The program can process input files as the format of `CSV` and `parquet` files.

### Running the software
`main.py` is the main entry for training and evaluating models. The behavior of the 
software depends on the configuration of the `config.ini` file.

The parameters and their settings are described below. The most important settings
are described first, subsequently the ones rarely changed:
#### Often changed
* workload (str) - The workload to run
  * base: Perform baseline experiments for rfc, svm and ann models 
  * mlm: Pretrain M-BERT using a masked label modeling task
  * los: Train M-BERT model for length of stay prediction
* task (str) - The prediction task so solve 
  * binary: Train M-BERT towards binary stratification 
  * category: Train M-BERT towards categorical classification
  * real: Train M-BERT as a regression model
* binary_thresh (int) - Threshold for binary classification
* categories (list) - Categories for categorical classification
* seq_hours (int) - Hours of sequences to use for classification
* load_mlm (boolean) - Load M-BERT using a pre-trained model
* save_model (boolean) - Save the trained model

#### Rarely changed
* years (list) - Years of data to include
* types (list) - Data types to include from sequences
* clip_los (int) - Clip the los of hospitalizations to x days
* use_logging (boolean) - Use Neptune logging or not
* experiment_name (str) - Experiment name logged to neptune AI
* neptune_project_id (str) - Name of Naptune AI project
* neptune_token_key (str) - Name of .env variable for storing neptune api token 
* lr (float) - Learning rate for training M-BERT
* warmup_proportion (float) - Warmup proportion for weight initialization
* weight_decay (float) - l2 regularization
* epochs (int) - Max epochs for training M-BERT
* batch_size (int) - Batch size for training M-BERT
* max_len_seq (int) - Maximum length of sequences
* use_gpu (boolean) - Use gpu for training or not
* patience (int) - Earlystopping training
* hidden_size (int) - Size of input token embeddings
* layer_dropout (float) - Dropout proportion after final layer of transformer encoder
* num_hidden_layers (int) - Num of hidden layers
* num_attention_heads (int) - Attention heads for each layer
* att_dropout (float) - Dropout proportion of attention in each layer
* intermediate_size (int) - Size of token embeddings for hidden layers
* hidden_act (str) - Activation function for transformer encoders
  * gelu
  * relu
* initializer_range (float) - Initializer range of parameter weight initialization
* features (list) - Embedding features to include in model
