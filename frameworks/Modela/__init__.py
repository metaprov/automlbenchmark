import os

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir, Namespace
from amlb import resources

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    data = dict(
        train=dict(
            X=dataset.train.X,
            y=dataset.train.y,
            data=dataset.train.data
        ),
        test=dict(
            X=dataset.test.X,
            y=dataset.test.y,
            data=dataset.test.data
        ),
        ml_task=dataset.type.name,
        target=dataset.target.name,
        trainers=Namespace.get(resources.config(), "trainers", 1)
    )

    if config.max_runtime_seconds <= 60:  # Modela needs at least 5-10 mins to process/train any dataset
        config.max_runtime_seconds = 600

    print(dataset.test.X, dataset.test.y.squeeze())

    env = os.environ.copy()
    del env['PATH']
    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config, options={'env':env})
