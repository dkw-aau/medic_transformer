# Example patient and data files
Currently, CSV and PARQUET formats are supported. 

### Patient file
Each row represents a patient hospitalization, with a sequence id linking to rows in the data file.

`patients.csv` should contain the following fields:

* `sequence_id (int)`- The id of the sequence
* `patient_id (int)`- The id of the patient
* `age (int)`- The age of the patient in years
* `sex (0/1)`- Binarized patient sex
* `hosp_start (datetime)`- Start date and time of the hospitalization
* `los (float)`- Length of stay in days

#### Example data

| sequence_id | patient_id | age | sex | hosp_start           | los  |
|-------------|------------|-----|-----|----------------------|------|
| 1           | 1          | 54  | 1   | 2017-02-19 15:23:00  | 3.96 |
| 2           | 2          | 57  | 0   | 2016-10-06 17:20:00  | 0.13 |
| 3           | 3          | 23  | 1   | 2017-01-20 18:33:00  | 2.20 |
| 4           | 3          | 27  | 1   | 2021-07-13 23:58:00  | 1.17 |
| 5           | 4          | 74  | 0   | 2016-01-03 17:50:00  | 3.15 |
| 6           | 4          | 79  | 0   | 2021-09-27 23:47:00  | 7.67 |

### Data file
Each row represents a medical event pertaining to a patient hospitalization

`data.csv` should contain the following fields:

* `token (str)`- The tokenized version of a medical event (e.g., Temp_Low)
* `token_orig (str)`- The original token before tokenization (e.g., Temp) 
* `event_time (datetime)`- Date and time of the event
* `event_type (str)`- The type of the event (e.g., Laboratory)
* `event_value (float)`- Original float/categorical value of event
* `sequence_id (int)`- The id of the sequence
`data.csv` specifies the EHR of a patient modelled as individual tokenized medical events

| token           | token_orig | event_time           | event_type | event_value | sequence_id |
|-----------------|------------|----------------------|------------|-------------|-------------|
| temp-low        | TEMP       | 2020-03-16 12:31:57  | vital      | 36.00       | 1           |
| puls-normal     | PULS       | 2020-03-16 12:31:57  | vital      | 55.00       | 1           |
| oxysat-normal   | OXYSAT     | 2020-03-16 12:31:57  | vital      | 100.00      | 1           |
| PROCZX500       | PROCZX500  | 2020-03-16 12:33:00  | procedure  | 1           | 1           |
| NPU21531-normal | NPU21531   | 2020-03-16 13:12:15  | labtest    | 4.80        | 1           |
| NPU03944-normal | NPU03944   | 2020-03-16 13:12:15  | labtest    | 0.90        | 1           |
| PROCZZ0149      | PROCZZ0149 | 2020-03-17 01:26:00  | procecure  | 1           | 2           |
| resp_normal     | RESP       | 2020-03-17 06:08:50  | vital      | 17.00       | 2           |
| puls_normal     | PULS       | 2020-03-17 06:08:50  | vital      | 81.00       | 2           |
| ATCR06AX13      | ATCR06AX13 | 2020-03-17 08:00:00  | medicine   | 1           | 2           |
