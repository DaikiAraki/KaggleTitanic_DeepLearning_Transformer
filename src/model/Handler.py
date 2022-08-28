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
from src.model.Model import Model

"""
モデルを保有し、またそれに対する入出力や学習などの処理を実行するクラス
モデルを使う際はこのクラスを通して使う
"""

class ModelHandler:

    def __init__(self, name="modelHandler"):
        """
        【注意】__init__()だけではモデルは構築されない。別途construct_model()を実行する必要がある。
        """
        self.name = name
        self.model = None


    def construct_model(self, initializer=None, name="model"):
        """
        モデルを構築する。
        initializerにInitValueクラスのインスタンスが与えられれば、それが持っている値を元にモデルを構築する。
        """
        self.model = Model(initValues=(initializer.get_initValues() if initializer is not None else None), name=name)


    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def run_inference(self, X):
        """
        推定モードで動作させる。
        単純なモデルの順方向処理の結果を出力する。
        :param X: [batch, width], 入力
        :return Y: [batch, width], 出力
        """
        L = tf.constant(False, shape=[])
        Y = self.model.call(X=X, L=L)  # [batch, width]
        return Y  # [batch, width]


    def run_inference_as_estimations(self, X):
        """
        推定モードで動作させる。
        Kaggle提出用のデータで出力する。
        :param X: [batch, width]
        :return estimations: [batch], np
        """
        Y = self.run_inference(X=X)  # [batch, width]
        estimations = np.argmax(Y.numpy(), axis=1)  # [batch], np
        return estimations  # [batch], np


    def run_training(self, X, Yt):
        """
        学習モードで動作させる。
        一度の呼び出して一回の学習が行われる。
        :param X: [batch, width], input
        :param Yt: [batch, width], target
        :return objective: [], loss in training
        """
        gradients, variables, objective = self.__getGradients(X=X, Yt=Yt)
        self.model.Optimizer.apply_gradients(zip(gradients, variables), name="Optimizer/proc/update")
        return objective  # []


    def run_verification(self, X, Yt):
        """
        評価モードで動作させる。
        与えられたX, Ytについて、現在のモデルでの誤差を出力する。
        :param X: [batch, width], input
        :param Yt: [batch, width], target
        :return objective: [], generalization loss
        """
        objective = self.__getObjectives(X=X, Yt=Yt)
        return objective  # []


    """ Get Gradients (for Training) """
    def __getGradients(self, X, Yt):
        """
        学習の為の勾配計算を行う。
        順方向出力とターゲットを用いて、誤差を計算し、そこからgradients（勾配）を得る。
        :param X: [batch, width], input
        :param Yt: [batch, width], target
        :return gradients, variables, objective: list, list, float
        """
        L = tf.constant(True, shape=[])
        with tf.GradientTape() as tape:
            Y = self.model.call(X=X, L=L)
            objective = tf.reduce_mean(self.model.ObjectiveProbability.call(Y=Y, Yt=Yt), keepdims=False)
            variables = self.model.get_trainable_variables()
            gradients = tape.gradient(objective, variables)
            return gradients, variables, objective


    def __getObjectives(self, X, Yt):
        """
        汎化能評価の為の誤差計算を行う。
        順方向出力とターゲットを用いて、誤差を得る。
        :param X: [batch, width], input
        :param Yt: [batch, width], target
        :return objective: [], loss
        """
        L = tf.constant(False, shape=[])
        Y = self.model.call(X=X, L=L)
        objective = tf.reduce_mean(self.model.ObjectiveProbability.call(Y=Y, Yt=Yt), keepdims=False)
        return objective


    def get_training_step(self):
        """
        :return: 現在の学習回数 (= iterations)
        """
        return self.model.Optimizer.iterations.numpy()



