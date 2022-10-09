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

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from src.model.Optimizer import Optimzier

"""
AdaBelief (https://arxiv.org/abs/2010.07468) を実装したクラス。
src.model.OptimizerAdam.Adamは、kerasのAdam（確かCかGOによるlibrary）をそのまま使っているが、
これはpythonで書いた(下記gen_ops_apply_AdaBelief())ため速度が劣る。
slotsの入出力が可能である。

### CAUTION ###
the all Variables names (tf.Variable._shared_name) must be unique in the model.
the all Variables names must not be named like "xxx_12". do not use "_"+number at the end of the name.
"""

class AdaBelief(Optimzier):

    _HAS_AGGREGATE_GRAD = True

    def __init__(self, learning_rate=1.e-4, momentum=0.9, rmsprop=0.999, epsilon=1.e-14, name="AdaBelief", **kwargs):
        super(AdaBelief, self).__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", momentum)
        self._set_hyper("beta_2", rmsprop)
        self.epsilon = epsilon or backend_config.epsilon()


    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 's')


    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaBelief, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = apply_state[(var_device, var_dtype)]["lr_t"] * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
        apply_state[(var_device, var_dtype)].update(
            dict(lr=lr,
                 epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                 beta_1_t=beta_1_t,
                 beta_1_power=beta_1_power,
                 one_minus_beta_1_t=1 - beta_1_t,
                 beta_2_t=beta_2_t,
                 beta_2_power=beta_2_power,
                 one_minus_beta_2_t=1 - beta_2_t,))


    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        m = self.get_slot(var, 'm')
        s = self.get_slot(var, 's')

        return gen_ops_apply_AdaBelief(var=var,
                                       m=m,
                                       s=s,
                                       beta1_power=tf.convert_to_tensor(coefficients['beta_1_power'], dtype=tf.float32),
                                       beta2_power=tf.convert_to_tensor(coefficients['beta_2_power'], dtype=tf.float32),
                                       lr=tf.convert_to_tensor(coefficients['lr_t'], dtype=tf.float32),
                                       beta1=tf.convert_to_tensor(coefficients['beta_1_t'], dtype=tf.float32),
                                       beta2=tf.convert_to_tensor(coefficients['beta_2_t'], dtype=tf.float32),
                                       epsilon=tf.convert_to_tensor(coefficients['epsilon'], dtype=tf.float32),
                                       grad=grad,
                                       use_locking=tf.convert_to_tensor(self._use_locking, dtype=tf.bool))


    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        m = self.get_slot(var, 'm')
        s = self.get_slot(var, 's')

        # op_m = beta1 * m + (1 - beta1) * g_t
        op_m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        op_m = state_ops.assign(ref=m, value=m * coefficients['beta_1_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([op_m]):
            op_m = self._resource_scatter_add(x=m, i=indices, v=op_m_scaled_g_values)

        # op_s = beta2 * s + (1 - beta2) * ((grad - op_m) ^ 2)
        s_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        op_s = state_ops.assign(ref=s, value=s * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([op_s]):
            op_s = self._resource_scatter_add(x=s, i=indices, v=s_scaled_g_values)

        op_s_sqrt = math_ops.sqrt(op_s)
        op_var = state_ops.assign_sub(ref=var, value=coefficients['lr'] * op_m / (op_s_sqrt + coefficients['epsilon']),
                                      use_locking=self._use_locking)
        updates = [op_var, op_m, op_s]
        return control_flow_ops.group(*updates)


    def get_config(self):
        config = super(AdaBelief, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
        })
        return config


@tf.function(experimental_relax_shapes=True)
def gen_ops_apply_AdaBelief(var, m, s, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking):
    op_m = state_ops.assign(ref=m,
                            value=beta1 * m + (1. - beta1) * grad,
                            use_locking=use_locking)
    op_s = state_ops.assign(ref=s,
                            value=beta2 * s + (1. - beta2) * math_ops.square(grad - op_m),
                            use_locking=use_locking)
    op_m_hat = math_ops.truediv(x=op_m, y=1. - beta1_power)
    op_s_hat = math_ops.truediv(x=op_s, y=1. - beta2_power)
    op_var = state_ops.assign_sub(ref=var,
                                  value=math_ops.multiply(x=op_m_hat, y=lr / (math_ops.sqrt(x=op_s_hat) + epsilon)),
                                  use_locking=use_locking)
    return op_var


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



