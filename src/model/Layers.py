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

from src.model.ModelCore import ModelLayer
from src.function.model_functions import *

"""
最も単純な一種類の層処理のみを持つクラス
weightやbiasなどのVariablesを有し、call()実行によって順方向出力を返す（微分計算の為には、呼び出し元でGradTapeを使う）
"""

class SelfAttentionLayer(ModelLayer):
    # shape of x = [batch, inWidth, inChannel]
    # shape of y = [batch, outWidth, outChannel]
    def __init__(self, headNum, nodeQ, nodeKV, nodeO, inWidth, inChl,
                 initValues=None, name="selfAttention"):
        """
        self-attention layer（multi-head）
        :param headNum: int, number of attention-heads
        :param nodeQ: int, number of output nodes of query projection
        :param nodeKV: int, number of output nodes of key and value projection
        :param nodeO: int, number of output nodes of attention layer
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValuesを持つ,
                           このブロックのinitValuesは、variableのnameをkeyにしてvariable初期化用の行列（ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(SelfAttentionLayer, self).__init__(name=name)
        n_wQ = "w_q"
        n_wK = "w_k"
        n_wV = "w_v"
        n_wO = "w_o"
        n_bO = "b_o"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        init_wQ = initVal[n_wQ] if (initVal is not None) and (n_wQ in initVal.keys()) else None
        init_wK = initVal[n_wK] if (initVal is not None) and (n_wK in initVal.keys()) else None
        init_wV = initVal[n_wV] if (initVal is not None) and (n_wV in initVal.keys()) else None
        init_wO = initVal[n_wO] if (initVal is not None) and (n_wO in initVal.keys()) else None
        init_bO = initVal[n_bO] if (initVal is not None) and (n_bO in initVal.keys()) else None
        self.headNum = headNum
        self.nodeQ = nodeQ
        self.nodeKV = nodeKV
        self.nodeO = nodeO
        self.outWidth = nodeO
        self.outChl = inChl

        with tf.name_scope(self.name):
            self.dim_q = tf.convert_to_tensor(nodeQ, dtype=tf.float32, name="dim_q")
            self.wQ = weight_variable([headNum, inWidth, nodeQ], initMatrix=init_wQ, name=n_wQ)
            self.wK = weight_variable([headNum, inWidth, nodeKV], initMatrix=init_wK, name=n_wK)
            self.wV = weight_variable([headNum, inWidth, nodeKV], initMatrix=init_wV, name=n_wV)
            self.wO = weight_variable([headNum * nodeQ, nodeO], initMatrix=init_wO, name=n_wO)
            self.bO = bias_variable([1, nodeO, 1], initMatrix=init_bO, name=n_bO)
            self._register_objects({n_wQ: self.wQ,
                                    n_wK: self.wK,
                                    n_wV: self.wV,
                                    n_wO: self.wO,
                                    n_bO: self.bO})

        self.result_import(self.wQ.name, init_wQ is not None)
        self.result_import(self.wK.name, init_wK is not None)
        self.result_import(self.wV.name, init_wV is not None)
        self.result_import(self.wO.name, init_wO is not None)
        self.result_import(self.bO.name, init_bO is not None)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
    def call(self, x):
        """
        :param x: [batch, inputWidth, inputChannel]
        :return y: [batch, outputWidth, outputChannel]]
        """
        with tf.name_scope(self.name + "/proc"):
            Q = tf.transpose(a=tf.tensordot(a=x, b=self.wQ, axes=[1, 1]), perm=[0, 2, 3, 1])  # [b,head,wQ,c]
            K = tf.transpose(a=tf.tensordot(a=x, b=self.wK, axes=[1, 1]), perm=[0, 2, 3, 1])  # [b,head,wKV,c]
            V = tf.transpose(a=tf.tensordot(a=x, b=self.wV, axes=[1, 1]), perm=[0, 2, 3, 1])  # [b,head,wKV,c]
            QK = tf.reduce_sum(tf.multiply(x=tf.expand_dims(Q, axis=3),  # [b,head,wQ,1,c]
                                           y=tf.expand_dims(K, axis=2)),  # [b,head,1,wKV,c]
                               axis=-1, keepdims=False)  # [b,head,wQ,wKV]
            QK_div_dq = tf.divide(x=QK, y=self.dim_q)  # [b,head,wQ,wKV]
            softmax_QK = tf.nn.softmax(logits=QK_div_dq, axis=-1)  # [b,head,wQ,wKV]
            O = tf.reduce_sum(tf.multiply(x=tf.expand_dims(softmax_QK, axis=4),  # [b,head,wQ,wKV,1]
                                          y=tf.expand_dims(V, axis=2)),  # [b,head,1,wKV,c]
                              axis=-2, keepdims=False)  # [b,haed,wQ,c]
            O_concat = tf.reshape(O, shape=[-1, self.headNum * self.nodeQ, self.outChl])  # [b,head*wQ,c]
            y = tf.add(x=tf.transpose(a=tf.tensordot(a=O_concat, b=self.wO, axes=[1, 0]), perm=[0, 2, 1]),
                       y=self.bO)  # [b,nodeO,c]
        return y  # [b,w,c]


class FcLayer(ModelLayer):
    # shape of x = [batch, inNode]
    # shape of w = [inNode, outNode]
    # shape of b = [1, outNode]
    # shape of y = [batch, outNode]
    def __init__(self, inNode, outNode, use_bias=True, initValues=None, name="fc"):
        """
        fc(fully-connected) layer
        :param inNode: int, width of input
        :param outNode: int, width of output
        :param use_bias: bool, whether to use biases in fc
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValuesを持つ,
                           このブロックのinitValuesは、variableのnameをkeyにしてvariable初期化用の行列（ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(FcLayer, self).__init__(name=name)
        n_w = "w"
        n_b = "b"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        init_w = initVal[n_w] if (initVal is not None) and (n_w in initVal.keys()) else None
        init_b = initVal[n_b] if (initVal is not None) and (n_b in initVal.keys()) else None
        self.outNode = outNode

        with tf.name_scope(self.name):
            self.use_bias = tf.convert_to_tensor(use_bias, dtype=tf.bool, name="use_bias")
            self.w = weight_variable([inNode, outNode], initMatrix=init_w, name=n_w)
            if use_bias:
                self.b = bias_variable([1, outNode], initMatrix=init_b, name=n_b)
                self._register_objects({n_w: self.w,
                                        n_b: self.b})
            else:
                self.b = tf.constant(0., dtype=tf.float32, name=n_b)
                self._register_objects({n_w: self.w})

        self.result_import(self.w.name, init_w is not None)
        if use_bias:
            self.result_import(self.b.name, init_b is not None)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def call(self, x):
        """
        :param x: [batch, inputNode]
        :return y: [batch, outputNode]
        """
        with tf.name_scope(self.name + "/proc"):
            y = tf.tensordot(a=x, b=self.w, axes=[1, 0])
            y = tf.cond(self.use_bias, lambda: tf.add(x=y, y=self.b), lambda: y)
        return y  # [b,w]


class FcDepthwiseLayer(ModelLayer):
    # shape of x = [batch, inWidth, inChannel]
    # shape of w = [inWidth, outWidth]
    # shape of b = [1, outWidth]
    # shape of y = [batch, outWidth, outChannel]
    def __init__(self, inWidth, outWidth, use_bias=True, initValues=None, name="fcDepthwise"):
        """
        depth-wise fc layer
        apply fc for width direction with remaining channel direction independent of each other
        :param inWidth: int, width of input
        :param outWidth: int, width of output
        :param use_bias: bool, whether to use biases
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValuesを持つ,
                           このブロックのinitValuesは、variableのnameをkeyにしてvariable初期化用の行列（ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(FcDepthwiseLayer, self).__init__(name=name)
        n_w = "w"
        n_b = "b"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        init_w = initVal[n_w] if (initVal is not None) and (n_w in initVal.keys()) else None
        init_b = initVal[n_b] if (initVal is not None) and (n_b in initVal.keys()) else None
        self.outNode = outWidth

        with tf.name_scope(self.name):
            self.use_bias = tf.convert_to_tensor(use_bias, dtype=tf.bool, name="use_bias")
            self.w = weight_variable([inWidth, outWidth], initMatrix=init_w, name=n_w)
            if use_bias:
                self.b = bias_variable([1, outWidth, 1], initMatrix=init_b, name=n_b)
                self._register_objects({n_w: self.w,
                                        n_b: self.b})
            else:
                self.b = tf.constant(0., dtype=tf.float32, name=n_b)
                self._register_objects({n_w: self.w})

        self.result_import(self.w.name, init_w is not None)
        if use_bias:
            self.result_import(self.b.name, init_b is not None)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
    def call(self, x):
        """
        :param x: [batch, inputWidth, inputChannel]
        :return y: [batch, outputWidth, outputChannel]
        """
        with tf.name_scope(self.name + "/proc"):
            y = tf.tensordot(a=x, b=self.w, axes=[1, 0])  # [b,inCh,outWidth]
            y = tf.transpose(a=y, perm=[0, 2, 1])  # [b,outWidth,inChl]=[b,w,c]
            y = tf.cond(self.use_bias, lambda: tf.add(x=y, y=self.b), lambda: y)  # [b,w,c]
        return y  # [b,w,c]


class FcPointwiseLayer(ModelLayer):
    # shape of x = [batch, inWidth, inChannel]
    # shape of w = [1, inChl, outChl]
    # shape of b = [1, 1, outChl]
    # shape of y = [batch, outWidth, outChannel]
    def __init__(self, inWidth, inChl, outChl, use_bias=True, initValues=None, name="fcPointwise"):
        """
        point-wise fc layer
        apply fc for channel direction with remaining width direction independent of each other
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param outChl: int, channel number of output
        :param use_bias: bool, whether to use biases
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValuesを持つ,
                           このブロックのinitValuesは、variableのnameをkeyにしてvariable初期化用の行列（ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(FcPointwiseLayer, self).__init__(name=name)
        n_w = "w"
        n_b = "b"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        init_w = initVal[n_w] if (initVal is not None) and (n_w in initVal.keys()) else None
        init_b = initVal[n_b] if (initVal is not None) and (n_b in initVal.keys()) else None
        self.outWidth = inWidth
        self.outChl = outChl

        with tf.name_scope(self.name):
            self.use_bias = tf.convert_to_tensor(use_bias, dtype=tf.bool, name="use_bias")
            self.w = weight_variable([1, inChl, self.outChl], initMatrix=init_w, name=n_w)
            if use_bias:
                self.b = bias_variable([1, 1, self.outChl], initMatrix=init_b, name=n_b)
                self._register_objects({n_w: self.w,
                                        n_b: self.b})
            else:
                self.b = tf.constant(0., dtype=tf.float32, name=n_b)
                self._register_objects({n_w: self.w})

        self.result_import(self.w.name, init_w is not None)
        if use_bias:
            self.result_import(self.b.name, init_b is not None)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
    def call(self, x):
        """
        :param x: [batch, inputWidth, inputChannel]
        :return y: [batch, outputWidth, outputChannel]
        """
        with tf.name_scope(self.name + "/proc"):
            y = tf.nn.conv1d(input=x, filters=self.w, stride=1, padding="SAME", data_format="NWC")  # [b,w,outCh]
            y = tf.cond(self.use_bias, lambda: tf.add(x=y, y=self.b), lambda: y)  # [b,w,c]
        return y  # [b,w,c]


class FlattenLayer(ModelLayer):
    # shape of x = [batch, width, channel]
    # shape of y = [batch, outWidth]
    def __init__(self, inWidth, inChl, initValues=None, name="flatten"):
        """
        layer that flatten input shape
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValuesを持つ,
                           このブロックのinitValuesは、variableのnameをkeyにしてvariable初期化用の行列（ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(FlattenLayer, self).__init__(name=name)
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.outWidth = inWidth * inChl

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
    def call(self, x):
        """
        :param x: [batch, inputWidth, inputChannel]
        :return y: [batch, outputWidth, outputChannel]
        """
        with tf.name_scope(self.name + "/proc"):
            y = tf.reshape(x, [-1, self.outWidth])
        return y


class ActivationLayer(ModelLayer):
    
    def __init__(self, actFunc, initValues=None, name="activate"):
        """
        layer that apply an activation function
        :param actFunc: activation function
        :param initValues: 
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(ActivationLayer, self).__init__(name=name)
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.actFunc = actFunc

    @tf.function
    def call(self, x):
        with tf.name_scope(self.name + "/proc"):
            y = self.actFunc(x)
        return y


class DropoutLayer(ModelLayer):

    def __init__(self, rate, initValues=None, name="dropout"):
        """
        layer that apply the dropout
        :param rate: dropout rate
        :param initValues: 
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(DropoutLayer, self).__init__(name=name)
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.rate = rate

    @tf.function
    def call(self, x, L):
        """
        :param x: [any shape]
        :param L: [], whether in training
        :return y: [same shape as x]
        """
        with tf.name_scope(self.name + "/proc"):
            y = tf.nn.dropout(x=x, rate=tf.cond(pred=L, true_fn=lambda: self.rate, false_fn=lambda: 0.))
        return y


class LayerNormLayer(ModelLayer):
    # shape of x,y = [-1, (shape)]
    def __init__(self, shape, axis, initValues=None, name="layerNorm"):
        """
        layer normalization layer
        the directions of normalization are depend on the argument "axis"
        :param shape: shape of input, except element [0](size in batch direction)
        :param axis: directions of normalization, must not be 0, int or list or tuple
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValuesを持つ,
                           このブロックのinitValuesは、variableのnameをkeyにしてvariable初期化用の行列（ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(LayerNormLayer, self).__init__(name=name)
        n_beta = "beta"
        n_gamma = "gamma"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        init_beta = initVal[n_beta] if (initVal is not None) and (n_beta in initVal.keys()) else None
        init_gamma = initVal[n_gamma] if (initVal is not None) and (n_gamma in initVal.keys()) else None
        self.axis = axis
        self.shape = [1] + shape  # add batch dimension

        with tf.name_scope(self.name):
            self.beta, self.gamma = layer_norm_variable(shape,
                                                        initMatrix_beta=init_beta, initMatrix_gamma=init_gamma,
                                                        name_beta=n_beta, name_gamma=n_gamma)
            self._register_objects({n_beta: self.beta,
                                    n_gamma: self.gamma})

        self.result_import(self.beta.name, init_beta is not None)
        self.result_import(self.gamma.name, init_gamma is not None)

    @tf.function
    def call(self, x):
        with tf.name_scope(self.name + "/proc"):
            mu = tf.reduce_mean(input_tensor=x, axis=self.axis, keepdims=True)
            sigma = tf.sqrt(x=tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=x, y=mu)),
                                             axis=self.axis, keepdims=True))
            h = tf.math.divide_no_nan(x=tf.subtract(x=x, y=mu), y=sigma)
            y = tf.add(x=tf.multiply(x=h, y=self.gamma), y=self.beta)
        return y


class ReshapingLayer(ModelLayer):
    # shape of x = [-1, x.shape[1:]]
    # shape of y = [-1, self.shape[1:]]
    def __init__(self, shape, initValues=None, name="reshaping"):
        """
        layer that reshape input's shape
        :param shape: shape after the reshaping, except element [0](size in batch direction)
        :param initValues: 
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(ReshapingLayer, self).__init__(name=name)
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.shape = shape
    
    @tf.function
    def call(self, x):
        with tf.name_scope(self.name + "/proc"):
            y = tf.reshape(x, [-1] + self.shape)
        return y


class HighwayNetLayer(ModelLayer):
    # shape of x,y = [-1, (xShape)]
    def __init__(self, xShape, initValues=None, name="highway"):
        """
        layer that integrate the other layer's output (x in below) and the residual item (xRef in below)
        the mixing ratio is changing by trainable variable 't' (= Highway Net)
        :param xShape: shape of input, except element [0](size in batch direction)
        :param initValues: 
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(HighwayNetLayer, self).__init__(name=name)
        n_t = "t"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        init_t = initVal[n_t] if (initVal is not None) and (n_t in initVal.keys()) else None
        self.xShape = [1] + xShape   # add batch dimension

        with tf.name_scope(self.name):
            self.t = highwayNet_variable(self.xShape, initMatrix=init_t, name="t")
            self._register_objects({n_t: self.t})

        self.result_import(self.t.name, init_t is not None)

    @tf.function
    def call(self, x, xRef):
        """
        :param x: the layer's output
        :param xRef: the residual item (= input of the layer)
        :return y: [same shape as x]
        """
        with tf.name_scope(self.name + "/proc"):
            t_tanh = tf.divide(x=tf.add(x=tf.math.tanh(x=self.t), y=1.), y=2.)
            y = tf.add(tf.multiply(x, tf.subtract(1., t_tanh)), tf.multiply(xRef, t_tanh))
        return y



