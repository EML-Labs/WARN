# WARN: Early Warning of Atrial Fibrillation Using Deep Learning

This repository provides scripts and libraries for training a neural network model that can predict atrial fibrillation from RR intervals. Processing ECG data into RR intervals, generating recurrence plots (RP) and labelling them based on the presence of atrial fibrillation.

## Citation

Please cite the following work if you use this code or data:

Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J.
Early Warning of Atrial Fibrillation Using Deep Learning. Patterns, 2024.


## Data

The data folder contains a group of patients from the open-source [PAF Prediction Challenge Database](https://physionet.org/content/afpdb/1.0.0/) on PhysioNet. This dataset can be used to test predictions, with the labels included in the data folder.

- **Test set:** Available at [Zenodo](https://doi.org/10.5281/zenodo.10815811). Labels are included in the `data` folder.
- **Training and Validation sets:** Access can be requested from Xiaoyun Yang at yangxiaoyun321@126.com.

## Getting Started

### Dependencies

- Python 3.9.12
- TensorFlow 2.8
- MATLAB R2020a

Ensure you have the above dependencies installed before proceeding with executing the program scripts.

### Executing Program

1. **Data Processing (`data_processing.m`):**
   - Generates data for training and testing.
   - Processes ECG signals to generate recurrence plots (RPs) and labels them for atrial fibrillation presence.

2. **Model Training (`train.py`):**
   - Script for training the neural network model.

3. **Making Predictions (`main_predictions.py`):**
   - Performs predictions on test data and visualizes the results.

## Usage Example

To train the model and make predictions, follow these steps:

1. Prepare your data according to the instructions in the Data section.
2. Run the data processing script in MATLAB: data_processing.m
3. Train the model with the training script: python train.py
4. Perform predictions and generate plots: python main_predictions.py

## Testing

The `test.py` script allows you to run various tests for the WARN model using the `Manager` class. You can specify test types, preprocessing parameters, and batch settings via command-line arguments.

### Usage

```bash
python test.py --test_type <TEST_TYPE> [options]
```

### Arguments

| Argument               | Short | Type  | Default   | Description                                                                 |
|------------------------|-------|-------|-----------|-----------------------------------------------------------------------------|
| `--test_type`          | `-t`  | str   | **required** | Type of test to run. Available options: `mimic_perform_ecg`, `mimic_perform_ppg` |
| `--segment_size`       | `-s`  | int   | 30        | Segment size used in preprocessing                                           |
| `--overlap`            | `-o`  | int   | 5         | Overlap size for preprocessing                                              |
| `--quality_threshold`  | `-q`  | float | 0.8       | Quality threshold for preprocessing                                         |
| `--shuffle_data`       | `-f`  | bool  | True      | Whether to shuffle data during testing                                      |
| `--batch_size`         | `-b`  | int   | 32        | Batch size for testing                                                      |

### Examples

**Run ECG test with default parameters:**

```bash
python test.py -t mimic_perform_ecg
```
**Run PPG test with custom segment size and batch size:**

```bash
python test.py -t mimic_perform_ppg -s 60 -b 64
```

**Run ECG test with lower quality threshold:**

```bash
python test.py -t mimic_perform_ecg -q 0.7
```

### Notes

- The `Manager` class handles initializing the selected test class (`MIMICPerformECGTest` or `MIMICPerformPPGTest`) and executing the test using the provided arguments.

- Only the test types defined in `Configurations.Types.TestTypes` are supported. Using any other value will raise a    `ValueError`.

- Boolean flags like  `--shuffle_data` can be set to `False` by passing `-f False`.