import json
import os
from os.path import join

import matplotlib.pyplot as plt
import smart_open as sopen

from genre_classification.data_model.evaluation import EvaluationMetrics


def save_metrics_to_json(file_to_store: str, metrics: EvaluationMetrics):
    with sopen.open(file_to_store, 'w') as fout:
        json.dump(metrics.json(), fout)


def plot_metrics(file_with_metrics: str, path_to_save_plot: str):
    os.makedirs(path_to_save_plot, exist_ok=True)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    with sopen.open(file_with_metrics) as fin:
        fig.savefig(join(path_to_save_plot, f'{file_with_metrics.split(".")[0]}_plot.png'))
        plt.close()
