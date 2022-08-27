# Copyright 2022 Daiki Araki. All Rights Reserved.
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
# ==============================================================================

from pathlib import Path

"""
設定値
モデルそのものの変更が必要にならないもののみ
"""

class Config:

    def __init__(self):

        self.miniBatch_size = 128
        self.verify_num = 100

        self.path_working_dir = None
        self.path_model_data = None
        self.path_slots = None
        self.path_slots_log = None
        self.path_objective = None
        self.path_objective_verification = None
        self.set_path_working_dir(path_working_dir=Path(str(Path.cwd().parent) + "\\data"))


    def set_path_working_dir(self, path_working_dir):
        self.path_working_dir = Path(str(path_working_dir))
        self.path_working_dir.mkdir(parents=True, exist_ok=True)
        self.path_model_data = Path(str(path_working_dir) + "\\model")  # for model data
        self.path_slots = Path(str(path_working_dir) + "\\slots")  # for optimizer's slots
        self.path_slots_log = Path(str(path_working_dir) + "\\import_slots.log")
        self.path_objective = Path(str(path_working_dir) + "\\minimizing_objective.csv")
        self.path_objective_verification = Path(str(path_working_dir) + "\\minimizing_objective_verification.csv")

    def set_miniBatch_size(self, size):
        self.miniBatch_size = size

    def set_verify(self, num):
        self.verify_num = num



