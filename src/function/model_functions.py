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

import math
import tensorflow as tf

"""
tensorflow の tesnor に対して用いる functions
"""

""" activation functions """

def mish(x):  # mish関数
    return tf.multiply(x=x, y=tf.tanh(x=tf.math.softplus(features=x)))

def leakyRelu(x, coef=1.e-3):  # leaky-ReLU関数
    return tf.nn.leaky_relu(features=x, alpha=coef)

def elu(x):  # ELU関数 (Exponential Linear Units)
    return tf.nn.elu(features=x)

def eluPlus(x, c=1.):  # 出力が正領域のみになるようにシフトしたELU関数
    return tf.divide(x=tf.add(x=tf.nn.elu(tf.subtract(x=tf.multiply(x=x, y=c), y=1.)), y=1.), y=c)

def identity(x):  # y = x
    return x

def softmax(x):  # softmax関数
    return tf.nn.softmax(logits=x)

def sigmoid(x):  # sigmoid関数
    return tf.nn.sigmoid(x=x)

def tanh(x):  # tanh関数
    return tf.nn.tanh(x=x)

def gaussianFunction(x, c, dev):  # ガウス関数
    y = tf.exp((-1. / 2.) * tf.pow((x - c) / dev, 2.)) / (tf.sqrt(2. * math.pi) * dev)
    return y

# tf.Variable として weight を作成・初期化して返す関数
def weight_variable(shape, init_stddev=1.e-1, initMatrix=None, name="w"):
    initArr = (tf.random.truncated_normal(shape, mean=0.0, stddev=init_stddev)
               if ((initMatrix is None) or (list(initMatrix.shape) != shape))
               else initMatrix)
    w = tf.Variable(initArr, dtype=tf.float32, name=name)
    return w

# tf.Varaible として bias を作成・初期化して返す関数
def bias_variable(shape, init_value=0., initMatrix=None, name="b"):
    initArr = (tf.constant(init_value, shape=shape)
               if ((initMatrix is None) or (list(initMatrix.shape) != shape)) else
               initMatrix)
    b = tf.Variable(initArr, dtype=tf.float32, name=name)
    return b

# highwayNet の内部変数 t を tf.Variable として作成・初期化して返す関数
def highwayNet_variable(shape, init_val=1.e-1, initMatrix=None, name="t"):
    with tf.name_scope(name):
        initArr = (tf.constant( init_val, dtype=tf.float32, shape=shape )
                   if ((initMatrix is None) or (list(initMatrix.shape) != shape))
                   else initMatrix)
        t = tf.Variable(initArr, dtype=tf.float32, name=name)
    return t

# layer-normalization の内部変数 beta, gamma を tf.Variable として作成・初期化して返す関数
def layer_norm_variable(shape, initMatrix_beta=None, initMatrix_gamma=None,
                        name_beta="beta", name_gamma="gamma"):
    initArr_beta    = (tf.constant(0., dtype=tf.float32, shape=shape)
                       if ((initMatrix_beta is None) or (list(initMatrix_beta.shape) != shape))
                       else initMatrix_beta)
    initArr_gamma   = (tf.constant(1., dtype=tf.float32, shape=shape)
                       if ((initMatrix_gamma is None) or (list(initMatrix_gamma.shape) != shape))
                       else initMatrix_gamma)
    beta            = tf.Variable(initArr_beta, dtype=tf.float32, name=name_beta)
    gamma           = tf.Variable(initArr_gamma, dtype=tf.float32, name=name_gamma)
    return beta, gamma

# SMA (simple moving average) を計算する（長さは - (n - 1) される）
def sma1d_valid(x, n):  # shape of x = [batch, width, 1] (NWC)
    filt = tf.constant(value=1./n, shape=[n, 1, 1])  # [filterWidth, inputChannel, outputChannel]
    y = tf.nn.conv1d(input=x, filters=filt, stride=1, padding="VALID", data_format="NWC")
    return y

# SMA を計算する（端を padding するので長さは入力と同じ）
def sma1d_same(x, n):  # shape of x = [batch, width, 1] (NWC)
    filt    = tf.constant(value=1./n, shape=[n, 1, 1])  # [filterWidth, inputChannel, outputChannel]
    y       = tf.nn.conv1d(input=x, filters=filt, stride=1, padding="SAME", data_format="NWC")
    return y

# 一階差分の計算（５次元Tensorまで可）
def tensorDiff(x, axis):

    if axis > 5:
        print("ERROR: function tensorDiff() is not able to difference of the axis bigger than 6."
              + " check the source code.")

    xNext   = tf.roll(x, -1, axis=axis)
    diff    = tf.subtract(xNext, x)
    diff    = (diff[:-1] if axis == 0 else
              (diff[:, :-1] if axis == 1 else
              (diff[:, :, :-1] if axis == 2 else
              (diff[:, :, :, :-1] if axis == 3 else
              (diff[:, :, :, :, :-1] if axis == 4 else
              (diff[:, :, :, :, :, :-1] if axis == 5 else diff))))))
    return diff



