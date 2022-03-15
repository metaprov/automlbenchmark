import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import closing
import socket

import modela
import numpy as np
import pandas as pd
from modela import TaskType, Metric, Modela, ModelSearch, Training, Study, DatasetPhase, StudyPhase, Model, ModelPhase, \
    Workload

from amlb import AutoMLError
from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** Modela SDK [v{modela.__version__}] ****\n")
    API: Modela = None
    try:
        API = Modela(port_forward=True, username="admin", password="admin")
    except Exception as ex:
        log.error("Unable to connect to API gateway: %s", ex)

    task_type = dict(
        binary=TaskType.BinaryClassification,
        multiclass=TaskType.MultiClassification,
        regression=TaskType.Regression,
    )
    task = task_type.get(dataset.ml_task)

    metrics_map = dict(
        binary=dict(
            auc=Metric.RocAuc,
            accuracy=Metric.Accuracy,
            default=Metric.Accuracy
        ),
        multiclass=dict(
            auc=Metric.RocAuc,
            accuracy=Metric.Accuracy,
            default=Metric.Accuracy,
        ),
        regression=dict(
            logloss=Metric.LogLoss,
            rmse=Metric.RMSE,
        )
    )

    eval_metric = metrics_map.get(dataset.ml_task).get(config.metric) \
        if config.metric in metrics_map.get(dataset.ml_task) else metrics_map.get(dataset.ml_task)["default"]


    # TODO: create separate datasets for train/test
    study: Study = None
    model: Model = None
    with Timer() as training:
        md_dataset = API.Dataset("iris-product", "automl11-%s" % config["name"],
                                 gen_datasource=True,
                                 target_column=dataset.target,
                                 task_type=task,
                                 bucket="default-minio-bucket",
                                 workload=Workload("general-large"),
                                 dataframe=pd.concat([dataset.train.data, dataset.test.data]))

        md_dataset.submit(replace=True)
        md_dataset.visualize(show_progress_bar=False)
        study = API.Study("iris-product", "automl-%s" % config["name"],
                          dataset=md_dataset,
                          task_type=task,
                          objective=eval_metric,
                          bucket="default-minio-bucket",
                          search=ModelSearch(MaxModels=2, Trainers=dataset.trainers),
                          trainer_template=Training(EarlyStop=True))
        study.submit(replace=True)
        study.visualize(show_progress_bar=False)


    predictions = np.ndarray(0, dtype=str)
    with Timer() as predict:
        predictor = API.Predictor("iris-product", "automl-benchmark",
                                  model=study.best_model)
        predictor.submit(replace=True)
        predictor.wait_until_ready()
        for _, row in dataset.test.X.iterrows():
            prediction = predictor.predict([{col: row[col] for col in dataset.test.X.columns}])[0]
            predictions = np.append(predictions, prediction.Label)
            print(prediction)

    predictions, probabilities, probabilities_labels = None, None, None

    return result(
        output_file=config.output_predictions_file,
        predictions=predictions,
        truth=dataset.test.y.squeeze(),
        probabilities=probabilities,
        probabilities_labels=probabilities_labels,
        models_count=study.best_model.spec.TrialID,
        training_duration=training.duration,
        predict_duration=predict.duration
    )


if __name__ == "__main__":
    call_run(run)
