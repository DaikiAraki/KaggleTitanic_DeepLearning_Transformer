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

import tensorflow as tf

"""
最小化目標（誤差）を計算するクラス（インスタンス化して使用）
"""

class ObjectiveProbability:

    def __init__(self, name="ObjectiveProbability"):
        """
        :param name: str
        """
        self.name = name
        with tf.name_scope(self.name):
            pass

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def call(self, Y, Yt):
        """
        :param Y: [b,w], model output
        :param Yt: [b,w], target
        :return objective: [batch]
        """
        with tf.name_scope(self.name + "/proc"):
            objective = tf.sqrt(
                x=tf.reduce_sum(
                    input_tensor=tf.square(
                        x=tf.subtract(x=Y, y=Yt)),
                    axis=1, keepdims=False))  # Root Mean Squared Error, [b]

        return objective  # [b]



