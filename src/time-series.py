import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import json
import os
from itertools import islice
from pathlib import Path
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions

mx.random.seed(0)
np.random.seed(0)

print(f"Available datasets: {list(dataset_recipes.keys())}")

dataset = get_dataset("m4_daily", regenerate=True)

train_entry = next(iter(dataset.train))
train_entry.keys()

test_entry = next(iter(dataset.test))
test_entry.keys()
test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

print(
    f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}"
)
print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.freq}")

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=2 * dataset.metadata.prediction_length,
    freq=dataset.metadata.freq,
    trainer=Trainer(
        ctx="gpu",
        epochs=15,
        learning_rate=1e-3,
        hybridize=True,
        num_batches_per_epoch=100,
    ),
)

predictor = estimator.train(dataset.train)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)

forecast_entry = forecasts[0]

print(f"Number of sample paths: {forecast_entry.num_samples}")
print(f"Dimension of samples: {forecast_entry.samples.shape}")
print(f"Start date of the forecast window: {forecast_entry.start_date}")
print(f"Frequency of the time series: {forecast_entry.freq}")

print(f"Mean of the future window:\n {forecast_entry.mean}")
print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")
