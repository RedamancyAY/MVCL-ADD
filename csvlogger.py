# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import os
from typing_extensions import override
from pytorch_lightning.loggers.csv_logs import ExperimentWriter, CSVLogger
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.rank_zero import rank_zero_warn

class CustomNameExperimentWriter(ExperimentWriter):
    r"""Experiment writer for CSVLogger.

    Currently, supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    This logger supports logging to remote filesystems via ``fsspec``. Make sure you have it installed.

    继承自：
    1. https://github.com/Lightning-AI/pytorch-lightning/blob/3740546899aedad77c80db6b57f194e68c455e28/src/lightning/pytorch/loggers/csv_logs.py#L41
    2. https://github.com/Lightning-AI/pytorch-lightning/blob/3740546899aedad77c80db6b57f194e68c455e28/src/lightning/fabric/loggers/csv_logs.py#L193
    Args:
        log_dir: Directory for the experiment logs

    """
    
    # def __init__(self, log_dir: str, csv_name=None) -> None:
    #     super().__init__(log_dir=log_dir)

    #     self.metrics_file_path = os.path.join(self.log_dir, csv_name or 'metrics.csv')


    def __init__(self, log_dir: str, csv_name=None) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        # self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)
        self.metrics_file_path = os.path.join(self.log_dir, csv_name or 'metrics.csv') # only change this line


        
        print('hello', self.metrics_file_path)
        self._check_log_dir_exists(self.metrics_file_path)
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self.hparams: Dict[str, Any] = {}
    

    def _record_new_keys(self):
        """Records new keys that have not been logged before.
        
            copy from https://github.com/Lightning-AI/pytorch-lightning/blob/75e112f138ec5cdd8eee2c26720b480719f415d6/src/lightning/fabric/loggers/csv_logs.py#L251
            this code can sort the columns for the csv file.
        
        """
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()

        
        return new_keys

    @override
    def _check_log_dir_exists(self, s) -> None:
        print(s)
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
            print('hello', self.metrics_file_path)
            if self._fs.isfile(self.metrics_file_path):
                self._fs.rm_file(self.metrics_file_path)
                


class CustomNameCSVLogger(CSVLogger):
    
    def __init__(
        self,
        *args, csv_name = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._csv_name = csv_name

        print('hello', csv_name)
    
    
    @property
    @override
    @rank_zero_experiment
    def experiment(self) -> CustomNameExperimentWriter:
        r"""Actual _ExperimentWriter object. To use _ExperimentWriter features in your
        :class:`~lightning.pytorch.core.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = CustomNameExperimentWriter(log_dir=self.log_dir, csv_name=self._csv_name)
        return self._experiment