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
import tensorflow as tf

def argmax(a):

    # 引数の条件：１次元の list か tuple か np.ndarray である事
    assert type(a) in [list, tuple, np.ndarray] and (not any(hasattr(a_i, "__iter__") for a_i in a))

    if isinstance(a, np.ndarray):
        return np.argmax(a=a)

    else:
        return a[a.index(max(a))]


def argmin(a):

    # 引数の条件：１次元の list か tuple か ndarray である事
    assert type(a) in [list, tuple, np.ndarray] and (not any(hasattr(a_i, "__iter__") for a_i in a))

    if isinstance(a, np.ndarray):
        return np.argmin(a=a)

    else:
        return a[a.index(min(a))]


def sign(v):

    calc = lambda x: type(x)(1 if (x > 0) else (-1 if (x < 0) else 0))

    if type(v) in (list, tuple):
        obj_tp = type(v)
        return obj_tp((calc(x=x) for x in v))

    elif isinstance(v, np.ndarray):
        return np.sign(x=v)

    elif type(v) in (int, float):
        return calc(v)

    else:
        raise Exception(
            "ERROR: invalid argument type '" + str(type(v)) + "' in general_func.sign()\n"
            "type of argument 'v' must be in (int, float, list, tuple, np.ndarray)")


def minabs(a):

    # 引数の条件：１次元の list か tuple か ndarray
    assert (type(a) in (list, tuple, np.ndarray)) and (not any(hasattr(a_i, "__iter__") for a_i in a))

    return min((abs(v) for v in a))


def flatten(data):
    # list か tuple について、flattenして返す
    return [
        b for a in data for b in (  # 二重forなので記述順に注意
            flatten(a) if (hasattr(a, "__iter__") and (not isinstance(a, str)) and (not isinstance(a, tf.Variable)))
            else (a,))]



