import asyncio
import json
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
from modela import (
    TaskType, Metric, Modela, ModelSearch, Training, EarlyStopping,
    Study, DatasetPhase, StudyPhase, Model, ModelPhase, \
    Workload, DataSplit, DataSplitMethod, FeatureEngineeringSearch, Ensemble, Predictor, ClassicEstimator
)

from amlb import AutoMLError
from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** Modela SDK [v{modela.__version__}] ****\n")
    API: Modela = None
    try:
        API = Modela(port_forward=True, username="admin", password="admin", tenant="modela")
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

    estimator_map = dict(
        binary=ClassicEstimator.XGBClassifier,
        multiclass=ClassicEstimator.XGBClassifier,
        regression=ClassicEstimator.XGBRegressor
    )

    eval_metric = metrics_map.get(dataset.ml_task).get(config.metric) \
        if config.metric in metrics_map.get(dataset.ml_task) else metrics_map.get(dataset.ml_task)["default"]

    config["name"] = config["name"].replace("_", "-", -1)
    predictor = API.Predictor("iris-product", "benchmark-%s" % config["name"].lower())
    if not predictor.default_resource:
        predictions, probabilities, probabilities_labels = make_prediction(dataset, predictor, config.type != "classification")
        return result(
            output_file=config.output_predictions_file,
            predictions=predictions,
            truth=dataset.test.y.squeeze(),
            probabilities=probabilities,
            probabilities_labels=probabilities_labels,
        )

    dataset.train.data.to_csv('OUT_DAT.csv', index=False)
    with Timer() as training:
        md_datasource = API.DataSource("iris-product", "automl-%s" % config["name"].lower(),
                                       infer_dataframe=dataset.test.data,
                                       target_column=dataset.target)
        md_datasource.submit(replace=True)
        md_train_dataset = API.Dataset("iris-product", "benchmark-train-%s" % config["name"].lower(),
                                       datasource=md_datasource,
                                       task_type=task,
                                       bucket="default-minio-bucket",
                                       workload=Workload("memory-xlarge"),
                                       dataframe=dataset.train.data,
                                       fast=True)

        md_test_dataset = API.Dataset("iris-product", "benchmark-test-%s" % config["name"].lower(),
                                      datasource=md_datasource,
                                      task_type=task,
                                      bucket="default-minio-bucket",
                                      workload=Workload("memory-xlarge"),
                                      dataframe=dataset.test.data,
                                      fast=True)

        if md_train_dataset.default_resource:  # Use already-existing resources if possible
            md_train_dataset.submit(replace=True)
        if md_test_dataset.default_resource:
            md_test_dataset.submit(replace=True)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(md_train_dataset.wait_until_phase(DatasetPhase.Ready))
        loop.run_until_complete(md_test_dataset.wait_until_phase(DatasetPhase.Ready))
        study = API.Study("iris-product", "benchmark-%s" % config["name"].lower(),
                          dataset=md_train_dataset.name,
                          objective=eval_metric,
                          bucket="default-minio-bucket",
                          search=ModelSearch(
                            MaxModels=100, Trainers=4,#dataset.trainers),
                            EarlyStop=EarlyStopping(Enabled=True, MinModelsWithNoProgress=20, Initial=50),
                            MinBestScore=10000
                          ),
                          fe_search=FeatureEngineeringSearch(
                            Enabled=True,
                            MaxModels=10,
                            MaxTrainers=5,
                            Estimator=estimator_map.get(dataset.ml_task)
                          ),
                          ensemble=Ensemble(
                             Enabled=True,
                             StackingEnsemble=True,
                             VotingEnsemble=True
                          ),
                          trainer_template=Training(
                              Split=DataSplit(TrainDataset=md_train_dataset.name,
                                              TestDataset=md_test_dataset.name,
                                              Method=DataSplitMethod.Auto),
                              Resources=Workload("compute-2xlarge"),
                              Seed=config.seed
                          ),
                          fast=True)

        if study.default_resource:
            study.submit(replace=True)
            study.visualize(show_progress_bar=False)
        else:
            study.visualize(show_progress_bar=False)
            loop.run_until_complete(study.wait_until_phase(StudyPhase.Completed))


    with Timer() as predict:
        predictor = API.Predictor("iris-product", "benchmark-%s" % config["name"].lower(),
                                  model=study.best_model.name)
        predictor.submit(replace=True)
        predictions, probabilities, probabilities_labels = make_prediction(dataset, predictor, config.type != "classification")


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

def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError

def make_prediction(dataset, predictor: Predictor, regression=False):
    predictions = np.ndarray(0, dtype=str)
    probabilities_labels, probabilities = None, None
    loop = asyncio.new_event_loop()
    loop.run_until_complete(predictor.wait_until_ready())
    for _, row in dataset.test.X.iterrows():
        prediction = predictor.predict(json.dumps([{col: row[col] for col in dataset.test.X.columns}], default=convert))
        prediction = prediction[0]
        predictions = np.append(predictions, prediction.Label)
        if regression:
            continue

        probability_vals = {prob.Label: prob.Probability for prob in prediction.Probabilities}
        if not probabilities_labels:
            probabilities_labels = [label for label in probability_vals.keys()]
            probabilities = np.zeros((0, len(probabilities_labels)), dtype=float)

        probabilities = np.vstack((probabilities, [probability_vals[label] for label in probabilities_labels]))

    return predictions, probabilities, probabilities_labels


if __name__ == "__main__":
    call_run(run)
