import json
import os

import smart_open as sopen
import torch

from genre_classification.data_model.evaluation import EvaluationMetrics


def save_model(checkpoint_path: str, model, model_name: str):
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, f'{model_name}_checkpoint.pt'))


def load_model(model_path: str, model, device):
    return model.load_state_dict(torch.load(model_path, map_location=device))


def save_metrics(file_name: str, metrics: EvaluationMetrics):
    with sopen.open(file_name, "w") as json_file:
        json.dump(metrics.dict(), json_file, indent=4)


def load_metric(file_name: str) -> EvaluationMetrics:
    with sopen.open(file_name, "r") as json_file:
        metrics = json.load(json_file)
    return EvaluationMetrics(**metrics)
