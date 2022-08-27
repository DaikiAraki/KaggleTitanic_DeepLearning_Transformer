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

import numpy as np
from pathlib import Path

"""
保存したモデルを読み込むためのクラス
読み込んで、そのデータをModelクラスの初期化パラメータの形状で吐き出す
pathをセットしてから、make_initValues()を呼び出すことで、モデルデータを読み出す
"""

class InitValue:

    def __init__(self, path=None):
        """
        :param path: pathlib.Path, path of the top folder of a saved model(saved by this system) [default: "\\model"]
        """
        assert isinstance(path, Path)
        self.path_root = path
        self.initValues = {}

    def set_path(self, path_root):
        self.path_root = path_root

    def get_initValues(self):
        return self.initValues  # parameters for initialization, (calling the make_initValues() are needed before this)

    def make_initValues(self):
        """
        read a model that is in self.path_root and keep it in self.initValues
        """
        if self.path_root is not None:
            self.initValues = InitValue.__readInitMatrixes__(self.path_root)

    @staticmethod
    def __readInitMatrixes__(path):

        if not path.exists():
            return None

        result_dict = {}
        paths = list(path.glob("*"))
        names = [str(paths[i].resolve())[len(str(path)):].replace("\\", "") for i in range(len(paths))]

        for (p, n) in zip(paths, names):

            if n[-4:] == ".npy":
                result_dict[n.replace(".npy", "")] = np.load(str(p), allow_pickle=True)

            elif not ("." in n):
                result_dict[n] = InitValue.__readInitMatrixes__(p)

            else:
                pass

        return result_dict



