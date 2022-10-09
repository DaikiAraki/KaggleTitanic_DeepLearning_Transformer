# Copyright 2022 Daiki Araki.
# And this file includes the works that are distributed in the Apache License 2.0 by 2018 The TensorFlow Authors.
# the works (methods) by 2018 The TensorFlow Authors are pointed by the phrase "Replicated" in comments.
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

# tensorflow v2.9.1

import abc
import functools
import six
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from src.function.io_log_slot_import import save_log_slot_import

"""
tensorflow.python.keras.optimizer_v2.optimizer_v2 クラスをカスタムし、
slots(optimizerで使うmoment類)のimportとexportをできるようにしたクラス

### CAUTION ###
the all Variables names (tf.Variable._shared_name) must be unique in the model
the all Variables names must not be named like "xxx_12". do not use "_"+number at the end of the name.
"""

@six.add_metaclass(abc.ABCMeta)
class Optimzier(optimizer_v2.OptimizerV2):

    def __init__(self, name="Optimizer", gradient_aggregator=None, gradient_transformers=None, **kwargs):
        super(Optimzier, self).__init__(name=name,
                                        gradient_aggregator=gradient_aggregator,
                                        gradient_transformers=gradient_transformers,
                                        **kwargs)
        self._slots_path = None
        self._slots_log_path = None


    def get_slots(self):
        return self._slots


    def set_slots_path(self, path, path_log):
        self._slots_path = path
        self._slots_log_path = path_log


    # Replicated and Editting from :
    #     https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/optimizer_v2/optimizer_v2.py
    # the changes are indicated by '[C]' symbols
    def add_slot(self, var, slot_name, initializer="zeros", shape=None):
        """Add a new slot variable for `var`.
        A slot variable is an additional variable associated with `var` to train.
        It is allocated and managed by optimizers, e.g. `Adam`.
        Args:
          var: a `Variable` object.
          slot_name: name of the slot variable.
          initializer: initializer of the slot variable
          shape: (Optional) shape of the slot variable. If not set, it will default
          to the shape of `var`.
        Returns:
          A slot variable.
        """
        if slot_name not in self._slot_names:
            self._slot_names.append(slot_name)
        var_key = _var_key(var)
        slot_dict = self._slots.setdefault(var_key, {})
        weight = slot_dict.get(slot_name, None)
        if weight is None:
            if isinstance(initializer, str) or callable(initializer):
                initializer = initializers.get(initializer)
                if isinstance(initializer, tf.__internal__.tracking
                        .CheckpointInitialValueCallable) or (shape is not None):
                    slot_shape = shape
                else:
                    slot_shape = var.shape
                initial_value = functools.partial(
                    initializer, shape=slot_shape, dtype=var.dtype)
            else:
                initial_value = initializer

            with self._distribution_strategy_scope():
                strategy = tf.distribute.get_strategy()
                if not strategy.extended.variable_created_in_scope(var):
                    raise ValueError(
                        "Trying to create optimizer slot variable under the scope for "
                        "tf.distribute.Strategy ({}), which is different from the scope "
                        "used for the original variable ({}). Make sure the slot "
                        "variables are created under the same strategy scope. This may "
                        "happen if you're restoring from a checkpoint outside the scope."
                        .format(strategy, var))

                with strategy.extended.colocate_vars_with(var):
                    weight = tf.Variable(
                        name="%s/%s" % (var._shared_name, slot_name),  # pylint: disable=protected-access
                        dtype=var.dtype,
                        trainable=False,
                        initial_value=initial_value)
            backend.track_variable(weight)
            slot_dict[slot_name] = weight
            self._restore_slot_variable(
                slot_name=slot_name, variable=var,
                slot_variable=weight)
            self._weights.append(weight)
            self._import_slots(weight=weight, shared_name=var._shared_name, slot_name=slot_name)  # [C]: restore slots
        return weight


    # Replicated and Editting from :
    #     https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/optimizer_v2/optimizer_v2.py
    # the changes are indicated by '[C]' symbols
    @property
    def iterations(self):
        """Variable. The number of training steps this Optimizer has run."""
        if self._iterations is None:
            with self._distribution_strategy_scope():
                self._iterations = self.add_weight(
                    "iter",
                    shape=[],
                    dtype=tf.int64,
                    trainable=False,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
            self._weights.append(self._iterations)
            self._import_iterations(self._iterations)  # [C]: restore iterations
        return self._iterations


    # Replicated and Editting from :
    #     https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/optimizer_v2/optimizer_v2.py
    # the changes are indicated by '[C]' symbols
    @iterations.setter
    def iterations(self, variable):
        if self._iterations is not None:
            raise RuntimeError("Cannot set `iterations` to a new Variable after "
                               "the Optimizer weights have been created. Here it is "
                               f"attempting to set `iterations` to {variable}.")
        self._iterations = variable
        self._weights.append(self._iterations)
        self._import_iterations(self._iterations)  # [C]: restore iterations


    def _import_slots(self, weight, shared_name, slot_name):
        path = Path(str(self._slots_path) + "\\" + shared_name.replace("/", "\\") + "\\" + slot_name + ".npy")
        if path.exists():
            weight.assign(np.load(str(path), allow_pickle=True).astype(np.float32))
            imported = True
        else:
            imported = False
        save_log_slot_import(var_key=shared_name, slot_name=slot_name, imported=imported, path=self._slots_log_path)


    def _import_iterations(self, iterations):
        iterations_name = "iterations"
        path = Path(str(self._slots_path) + "\\" + iterations_name + "\\" + iterations_name + ".npy")
        if path.exists():
            iterations.assign(np.load(str(path), allow_pickle=True).astype(np.float32))
            imported = True
        else:
            imported = False
        save_log_slot_import(var_key=iterations_name, slot_name=iterations_name,
                             imported=imported, path=self._slots_log_path)


    def export_slots(self):
        """
        only the optimizer of the main model have to be saved and restored
        target-net is going to be updated by another way
        """
        slots = self.get_slots()
        for (var_key, slot_dict) in slots.items():
            isUniqueId = (var_key.rfind("_") > 0) and (var_key[-1] in "0123456789")
            shared_name = var_key[:var_key.rfind("_")] if isUniqueId else var_key

            path_var = Path(str(self._slots_path) + "\\" + shared_name.replace("/", "\\"))
            path_var.mkdir(parents=True, exist_ok=True)

            for (slot_name, slot) in slot_dict.items():
                path_slot = Path(str(path_var) + "\\" + slot_name + ".npy")
                np.save(str(path_slot), slot.numpy())


    def export_iterations(self):
        iterations_name = "iterations"
        path_dir = Path(str(self._slots_path) + "\\" + iterations_name)
        path_dir.mkdir(parents=True, exist_ok=True)
        path_itr = Path(str(path_dir) + "\\" + iterations_name + ".npy")
        np.save(str(path_itr), self.iterations.numpy())


# Replicated from :
#     https://github.com/keras-team/keras/blob/v2.9.0/keras/optimizers/optimizer_v2/optimizer_v2.py
# for calling in Optimzier.add_slots()
def _var_key(var):
    """Key for representing a primary variable, for looking up slots.
    In graph mode the name is derived from the var shared name.
    In eager mode the name is derived from the var unique id.
    If distribution strategy exists, get the primary variable first.
    Args:
    var: the variable.
    Returns:
    the unique name of the variable.
    """
    # pylint: disable=protected-access
    # Get the distributed variable if it exists.
    if hasattr(var, "_distributed_container"):
        var = var._distributed_container()
    if getattr(var, "_in_graph_mode", False):
        return var._shared_name
    return var._unique_id



