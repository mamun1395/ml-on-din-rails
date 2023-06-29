# Machine Learning on Deutsches Institut für Normung (DIN) rails

Anomaly Detection using Machine Learning on Fieldbus. This is part of the Master thesis.

## Dataset

`dataset` folder contains two folders namely `raw_dataset` and `processed_dataset`.

## Getting Started

1. `data_visualization.ipynb` visualizes the raw datasets for network configuration 1 and network configuration 2.

2. `data_processing.ipynb` transforms raw dataset folders into one single file on which ML algorithms can be applied.


## Time taken

| Model  | Training     | Test      |
|--------|--------------|-----------|
| MLP    |       53.59s | 1m 41.59s |
| CNN    |    7m 24.16s | 1m 27.97s |
| C-LSTM | 2h 8m 57.78s | 24m 2.58s |
