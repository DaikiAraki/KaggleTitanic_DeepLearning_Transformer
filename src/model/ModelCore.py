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

# tensorflow v2.4.1

import numpy as np
from pathlib import Path
from src.function.general_funcs import flatten
from src.function.io_log_variable_import import save_log_variable_import

"""
モデルの各パーツの基底クラス
LayerならModelLayerを、BlockならModelBlockを、ModelならModelCoreを継承して作成する
"""

class ModelCore:

    def __init__(self, name):
        self.name = name
        self._objects = {}  # dict of model Objects and Variables that attributes this object
        self._variables = {}  # hierarchical dictionary of all (containing sub parts) variables
        self._trainable_variables = []  # flat list of all trainable variables (containing those in subparts)

    def _register_objects(self, obj_dict):
        """
        register sub objects owned by this object
        :param obj_dict: dict{obj_name: obj}
        """
        for (k, v) in obj_dict.items():
            # object
            if any([ModelCore == i for i in type(v).__mro__]):
                self._objects.update({k: v})
                self._variables.update({k: v.get_variables()})
                self._trainable_variables.append(v.get_trainable_variables())
            # variables
            else:
                self._objects.update({k: v})
                self._variables.update({k: v})
                if (v is not None) and v.trainable:
                    self._trainable_variables.append(v)
        self._trainable_variables = flatten(self._trainable_variables)

    def get_trainable_variables(self):
        """
        get all of the trainable variables for training
        :return: all of the trainable variables (containing sub object's attributes) (flat list)
        """
        return self._trainable_variables

    def get_objects(self):
        """
        get all of the model part that owned by this object
        :return: all of the instances of ModelCore()'s subclasses
        """
        return self._objects

    @staticmethod
    def result_import(name, initIsNotNone):
        """
        write a log file about whether an initializing parameter has been used
        :param name: full name (containing all of the upper hierarchies' object names)
        :param initIsNotNone: whether the init parameter was not None
        """
        save_log_variable_import(name=name, imported=initIsNotNone)

    def export_variables(self, path_obj_dir):
        """
        export current objects that are owned by this object
        it is exported as (a numpy.ndarray's binary file) if (it is tf.Variable) else (a folder)
        :param path_obj_dir: pathlib.Path object corresponds to this object
        """
        assert isinstance(path_obj_dir, Path)

        path_obj_dir.mkdir(parents=True, exist_ok=True)

        for (k, o) in self.get_objects().items():

            if any([i == ModelCore for i in type(o).__mro__]):
                name = o.name
                o.export_variables(Path(str(path_obj_dir) + "\\" + name))

            elif o is not None:
                name = o.name[o.name.rindex("/") + 1: o.name.rindex(":")]
                np.save(str(path_obj_dir) + "\\" + name, o.numpy())


class ModelLayer(ModelCore):

    def __init__(self, name):
        super(ModelLayer, self).__init__(name=name)


class ModelBlock(ModelCore):

    def __init__(self, name):
        super(ModelBlock, self).__init__(name=name)



