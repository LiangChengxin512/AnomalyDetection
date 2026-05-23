# Description
A multivariate time-series anomaly detection model using Transformer.
My Research tpoic is **Using artificial intelligence methods and the theory of critical events for critical transitions forecasting in complex systems**. This topic is to develop of a method that would allow detecting anomalies in time series. The following figure illustrates the anomaly prediction performance of the TranAD_E model using top-k aggregation on the Server Machine Dataset (SMD). The visualization provides a detailed, step-by-step breakdown of how the model processes multivariate time-series data to detect anomalies over a specific test time-step interval (from 15,750 to roughly 16,150).
![Anomaly prediction visualization of TranAD E with top-k aggregation on the SMD
dataset](./anomaly_prediction.png)
# My Environment
Apple M4 processor (a 10-core CPU, 10-core GPU, 24 GB unified memory), leveraging the Metal Performance Shaders (MPS) backend and the onboard 16-core Neural Engine for hardware-accelerated training and inference.
Python 3.10 in the Conda Environment.
## Environment Setup
```bash
pip install -r requirements.txt
```
# Dataset Preprocessing
Preprocess all datasets using the command. Datasets are in the .\data
```bash
python preprocess.py SMAP MSL SWaT WADI SMD MSDS UCR MBA NAB
```
# Train TranAD_E model
```python
python experiments/train_tranad_variants.py \
  --model TranAD_E --dataset SMD --epochs 10 --device auto \
  --score-agg topk --score-topk 3
```
# Train baseline model
```python
python experiments/train_tranad_variants.py \
  --model TranAD --dataset SMD --epochs 10 --device auto \
  --score-agg mean
```

# Evaluate specific model
```python
python experiments/test_tranad_variants.py \
  --model TranAD_E \
  --dataset SMAP \
  --batch-size 128 \
  --device auto \
  --checkpoint-dir experiment_checkpoints
```


# Result Reproduction
```bash
bash run_tranad_e_table2_benchmark.sh
```