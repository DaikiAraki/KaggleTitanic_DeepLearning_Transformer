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

# tensorflow v2.9.1

from src.model.ModelCore import ModelBlock
from src.model.Layers import *

"""
複数のLayerを含むクラス
TransformerやConvなどを実装する際に必要なDropoutやNormalizaionのLayerを統合したユニット
"""

class TransformerBlock(ModelBlock):
    # shape of x = [batch, inWidth, inChl]
    # shape of y = [batch, outWidth, outChl]
    def __init__(self, inWidth, inChl, outChl, qkvChl, actFunc, headNum=16,
                 innerExpansionRatio=4, downsampling=False, use_firstNormalization=True, use_secondNormalization=True,
                 initValues=None, name="transformerBlock"):
        """
        block that mainly apply self-attention
        like a ViT or CvT or CoAtNet, construct an MLP part after the self-attention part
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param outChl: int, channel number of output
        :param qkvChl: int, width of the query, key, value matrices of attention
        :param actFunc: activation function
        :param headNum: int, number of attention-heads
        :param innerExpansionRatio: int, inner expansion ratio of channel (Squeeze-and-Excitation)
        :param downsampling: whether applying the downsampling conv
        :param use_firstNormalization: bool, whether to use a normalization before self-attention
        :param use_secondNormalization: bool, whether to use a normalization before mlp part
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValues
                           （attrがVariableなら、key=nameに初期化用のnumpy.ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(TransformerBlock, self).__init__(name=name)
        n_SaNormalize = "saNormalize"
        n_SelfAttention = "selfAttention"
        n_SaRes_Fc_Pw = "saResFcPw"
        n_SaHighway = "saHighway"
        n_FcNormalize = "fcNormalize"
        n_Fc0_Pw = "fc0Pw"
        n_ActivFc0 = "activFc0"
        n_Fc1_Pw = "fc1Pw"
        n_FcHighway = "fcHighway"
        n_DownsampleConv = "downsampleConv"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.inWidth = inWidth
        self.inChl = inChl
        self.outWidth = inWidth // (2 if downsampling else 1)
        self.outChl = outChl
        self.use_resPrj = inChl != outChl
        chl_expansion = inChl * innerExpansionRatio

        with tf.name_scope(self.name):

            """ Self-Attention Part """
            # normalization
            self.SaNormalize = LayerNormalizationLayer(
                shape=[inWidth, inChl], axis=[1, 2], initValues=initVal, name=n_SaNormalize
                ) if use_firstNormalization else None

            # self-attention
            self.SelfAttention = SelfAttentionLayer(
                headNum=headNum, qkvChl=qkvChl, outChl=outChl, inWidth=inWidth, inChl=inChl,
                initValues=initVal, name=n_SelfAttention)

            # squeeze a width size of the residual item (because of width squeezing in the self-attention)
            self.SaRes_Fc_Pw = FcPointwiseLayer(
                inWidth=inWidth, inChl=inChl, outChl=outChl, use_bias=True,
                initValues=initVal, name=n_SaRes_Fc_Pw
                ) if self.use_resPrj else None

            # integrate the self-attention output and the residual item
            self.SaHighway = HighwayNetLayer(
                xShape=[inWidth, outChl], initValues=initVal, name=n_SaHighway)

            """ MLP(Pointwise) Part """
            # normalization
            self.FcNormalize = LayerNormalizationLayer(
                shape=[inWidth, outChl], axis=[1, 2], initValues=initVal, name=n_FcNormalize
                ) if use_secondNormalization else None

            # point-wise fc (mlp part1), Excitation in channel axis, (proposed in the "Squeeze-and-Excitatioon Network")
            self.Fc0_Pw = FcPointwiseLayer(inWidth=inWidth, inChl=outChl, outChl=chl_expansion, use_bias=True,
                                           initValues=initVal, name=n_Fc0_Pw)

            # activation function
            self.ActivFc0 = ActivationLayer(actFunc=actFunc, initValues=initVal, name=n_ActivFc0)

            # point-wise fc (mlp part2), Squeeze in channel axis
            self.Fc1_Pw = FcPointwiseLayer(inWidth=inWidth, inChl=chl_expansion, outChl=outChl, use_bias=True,
                                           initValues=initVal, name=n_Fc1_Pw)

            # integrate the mlp output and the residual item
            self.FcHighway = HighwayNetLayer(xShape=[inWidth, outChl], initValues=initVal, name=n_FcHighway)

            """ Downsampling Part """
            # downsampling by convolution with stride=2
            self.DownsampleConv = ConvLayer(
                filterWidth=3, filterNum=outChl, inWidth=inWidth, inChl=outChl, stride=2,
                initValues=initVal, name=n_DownsampleConv
                ) if downsampling else None

            """ register objects for calling by get_trainable_variables() and etc """
            self._register_objects({n_SaNormalize: self.SaNormalize,
                                    n_SelfAttention: self.SelfAttention,
                                    n_SaRes_Fc_Pw: self.SaRes_Fc_Pw,
                                    n_SaHighway: self.SaHighway,
                                    n_FcNormalize: self.FcNormalize,
                                    n_Fc0_Pw: self.Fc0_Pw,
                                    n_ActivFc0: self.ActivFc0,
                                    n_Fc1_Pw: self.Fc1_Pw,
                                    n_FcHighway: self.FcHighway,
                                    n_DownsampleConv: self.DownsampleConv})

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.bool),))
    def call(self, x, L):
        """
        :param x: [batch, inputWidth, inputChannel]
        :param L: [], whether in training
        :return y: [batch, outputWidth, outputChannel]
        """
        x_normalize = self.SaNormalize.call(x) if self.SaNormalize is not None else x
        h_sa = self.SelfAttention.call(x_normalize)
        h_saRes = self.SaRes_Fc_Pw.call(x) if self.use_resPrj else x
        h_saHighway = self.SaHighway.call(h_sa, h_saRes)
        h_fcNorm = self.FcNormalize.call(h_saHighway) if self.FcNormalize is not None else h_saHighway
        h_fc0 = self.ActivFc0.call(self.Fc0_Pw.call(h_fcNorm))
        h_fc1 = self.Fc1_Pw.call(h_fc0)
        h_fcHighway = self.FcHighway.call(h_fc1, h_saHighway)
        y = self.DownsampleConv.call(h_fcHighway) if self.DownsampleConv is not None else h_fcHighway
        return y  # [b,w,c]


class CoAtRelativeAttentionLayer(ModelLayer):
    # CAUTION: this unit has inner positional encoding structure by wConv
    # shape of x = [batch, inWidth, inChannel]
    # shape of y = [batch, outWidth, outChannel]
    def __init__(self, headNum, qkvChl, outChl, inWidth, inChl,
                 initValues=None, log_silence=False, name="coAtRelativeAttention"):
        super(CoAtRelativeAttentionLayer, self).__init__(name=name)
        n = {
            "wConv": "w_conv",
            "wQ": "w_q",
            "wK": "w_k",
            "wV": "w_v",
            "wO": "w_o",
            "bO": "b_o"}
        initVal = self.read_initval(initValues=initValues, name=name)
        init_results = {}
        self.headNum = headNum
        self.qkvChl = qkvChl
        self.inWidth = inWidth
        self.inChl = inChl
        self.outWidth = inWidth
        self.outChl = outChl

        with tf.name_scope(self.name):
            self.root_dim_qkv = tf.convert_to_tensor(math.sqrt(qkvChl), dtype=tf.float32, name="dim_qkv")
            self.wConv, init_results[n["wConv"]] = weight_variable(
                [headNum, (2 * inWidth) - 1], initMatrix=self.read_initval(initVal, n["wConv"]), name=n["wConv"])
            self.wQ, init_results[n["wQ"]] = weight_variable(
                [headNum, inChl, qkvChl], initMatrix=self.read_initval(initVal, n["wQ"]), name=n["wQ"])
            self.wK, init_results[n["wK"]] = weight_variable(
                [headNum, inChl, qkvChl], initMatrix=self.read_initval(initVal, n["wK"]), name=n["wK"])
            self.wV, init_results[n["wV"]] = weight_variable(
                [headNum, inChl, qkvChl], initMatrix=self.read_initval(initVal, n["wV"]), name=n["wV"])
            self.wO, init_results[n["wO"]] = weight_variable(
                [headNum * qkvChl, outChl], initMatrix=self.read_initval(initVal, n["wO"]), name=n["wO"])
            self.bO, init_results[n["bO"]] = bias_variable(
                [1, 1, outChl], initMatrix=self.read_initval(initVal, n["bO"]), name=n["bO"])
            self._register_objects({n["wConv"]: self.wConv,
                                    n["wQ"]: self.wQ,
                                    n["wK"]: self.wK,
                                    n["wV"]: self.wV,
                                    n["wO"]: self.wO,
                                    n["bO"]: self.bO})

        if not log_silence:
            self.export_init_result(names=n, init_results=init_results)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
    def call(self, x):
        with tf.name_scope(self.name):
            Q = tf.transpose(a=tf.tensordot(a=x, b=self.wQ, axes=[2, 1]), perm=[0, 2, 1, 3])  # [b,head,w,cQKV]
            K = tf.transpose(a=tf.tensordot(a=x, b=self.wK, axes=[2, 1]), perm=[0, 2, 1, 3])  # [b,head,w,cQKV]
            V = tf.transpose(a=tf.tensordot(a=x, b=self.wV, axes=[2, 1]), perm=[0, 2, 1, 3])  # [b,head,w,cQKV]
            QK = tf.reduce_sum(
                tf.multiply(
                    x=tf.expand_dims(Q, axis=3),  # [b,head,w,1,cQKV]
                    y=tf.expand_dims(K, axis=2)),  # [b,head,1,w,cQKV]
                axis=4, keepdims=False)  # [b,head,w,w]
            QK_div_dk = tf.divide(x=QK, y=self.root_dim_qkv)  # [b,head,w,w]
            wConv_Toeplitz = self._ToeplitzMatrix_v1(wConv=self.wConv)  # [head,w,w]
            QK_add_wConv = tf.add(x=QK_div_dk, y=tf.expand_dims(wConv_Toeplitz, axis=0))  # [b,head,w,w]
            softmax_QK_add_wConv = tf.nn.softmax(logits=QK_add_wConv, axis=-1)  # [b,head,w,w]
            O = tf.reduce_sum(
                tf.multiply(
                    x=tf.expand_dims(softmax_QK_add_wConv, axis=4),  # [b,head,w,w,1]
                    y=tf.expand_dims(V, axis=2)),  # [b,head,1,w,cQKV]
                axis=3, keepdims=False)  # [b,haed,w,cQKV]
            O_concat = tf.reshape(
                tf.transpose(O, perm=[0, 2, 1, 3]),
                shape=[-1, self.inWidth, self.headNum * self.qkvChl])  # [b,w,head*cQKV]
            y = tf.add(x=tf.tensordot(a=O_concat, b=self.wO, axes=[2, 0]), y=self.bO)  # [b,w,c]
        return y  # [b,w,c]

    @tf.function
    def _ToeplitzMatrix_v1(self, wConv):
        """
        ver1: make the Toeplitz matrix by tf.extract_volume_patches()
        :param wConv: [head, -(width-1), ..., 0, ..., width-1]
        :return wConv_Toeplitz: [head, width, width]
        """
        with tf.name_scope(self.name + "_ToeplitzMatrix_v1"):
            wConv_Toeplitz = tf.reverse(
                tf.extract_volume_patches(
                    tf.reshape(wConv, shape=[self.headNum, -1, 1, 1, 1]),
                    ksizes=[1, self.inWidth, 1, 1, 1],
                    strides=[1, 1, 1, 1, 1],
                    padding="VALID")[:, :, 0, 0],
                axis=[1])  # [head,w,w]
        return wConv_Toeplitz  # [head,w,w]

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def _ToeplitzMatrix_v2(self, wConv):
        """
        ver2: make the Toeplitz matrix by tf.linalg.LinearOperatorToeplitz()
        :param wConv: [head, -(width-1), ..., 0, ..., width-1]
        :return wConv_Toeplitz: [head, width, width]
        """
        with tf.name_scope(self.name + "_ToeplitzMatrix_v2"):
            wConv_Toeplitz_op = tf.linalg.LinearOperatorToeplitz(
                col=tf.reverse(wConv[:, :self.inWidth], axis=[1]), row=wConv[:, self.inWidth - 1:])
            wConv_Toeplitz = wConv_Toeplitz_op.to_dense()  # [head,w,w]
        return wConv_Toeplitz  # [head,w,w]
    
    
class FcBlock(ModelBlock):
    # shape of x = [batch, inNode]
    # shape of y = [batch, outNode]
    def __init__(self, inNode, outNode, actFunc, use_bias=True,
                 normalize=False, pre_normalize=False, dropout=False, d_rate=0.,
                 initValues=None, name="fcBlock"):
        """
        block that mainly apply the fc(= fully connected layer)
        :param inNode: int, width of input
        :param outNode: int, width of output
        :param actFunc: activation function
        :param use_bias: bool, whether to use biases in fc
        :param normalize: bool, whether to use the normalization
        :param pre_normalize: bool, whether to do it before fc if the normalization will be applied
        :param dropout: bool, whether to use the dropout
        :param d_rate: float, dropout ratio
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValues
                           （attrがVariableなら、key=nameに初期化用のnumpy.ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(FcBlock, self).__init__(name=name)
        n_Fc = "fc"
        n_Normalize = "normalize"
        n_Activation = "activ"
        n_Dropout = "dropout"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.preNormalize = pre_normalize
        self.outNode = outNode
        shape_norm = [self.outNode] if not self.preNormalize else [inNode]

        with tf.name_scope(self.name):
            self.Fc = FcLayer(inNode=inNode, outNode=outNode, use_bias=use_bias, initValues=initVal, name=n_Fc)
            self.Normalize = LayerNormalizationLayer(
                shape=shape_norm, axis=1, initValues=initVal, name=n_Normalize) if normalize else None
            self.Activation = ActivationLayer(actFunc=actFunc, initValues=initVal, name=n_Activation)
            self.Dropout = DropoutLayer(rate=d_rate, initValues=initVal, name=n_Dropout) if dropout else None
            self._register_objects({n_Fc: self.Fc,
                                    n_Normalize: self.Normalize,
                                    n_Activation: self.Activation,
                                    n_Dropout: self.Dropout})

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.bool),))
    def call(self, x, L):
        """
        :param x: [batch, inNode]
        :param L: [], whether in training
        :return y: [batch, outNode]
        """
        y = self.Normalize.call(x) if (self.Normalize is not None) and self.preNormalize else x
        y = self.Fc.call(y)
        y = self.Normalize.call(y) if (self.Normalize is not None) and (not self.preNormalize) else y
        y = self.Activation.call(y)
        y = self.Dropout.call(y, L) if (self.Dropout is not None) else y
        return y  # [b, w]


class FlattenBlock(FcBlock):
    # totalNode = width*chl
    # shape of x = [batch, width, chl]
    # shape of y = [batch, node]
    def __init__(self, inWidth, inChl, outNode, actFunc, use_bias=True, normalize=False, pre_normalize=False,
                 dropout=False, d_rate=0., initValues=None, name="flattenBlock"):
        """
        block that flatten input and apply fc
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param outNode: int, width of output (number of node in fc)
        :param actFunc: activation function
        :param use_bias: bool, whether to use biases in fc
        :param normalize: bool, whether to use the normalization
        :param pre_normalize: bool, whether to do it before fc if the normalization will be applied
        :param dropout: bool, whether to use the dropout
        :param d_rate: float, dropout ratio
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValues
                           （attrがVariableなら、key=nameに初期化用のnumpy.ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.totalNode = inWidth * inChl  # inputNode for fc
        super(FlattenBlock, self).__init__(
            inNode=self.totalNode, outNode=outNode, actFunc=actFunc, use_bias=use_bias,
            normalize=normalize, pre_normalize=pre_normalize, dropout=dropout, d_rate=d_rate,
            initValues=initValues, name=name)
        n_Flatten = "flatten"

        with tf.name_scope(self.name):
            self.Flatten = FlattenLayer(inWidth=inWidth, inChl=inChl, initValues=initVal, name=n_Flatten)
            self._register_objects({n_Flatten: self.Flatten})

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.bool),))
    def call(self, x, L):
        """
        :param x: [batch, width, channel]
        :param L: [], whether in training
        :return y: [batch, node]
        """
        xf = self.Flatten.call(x=x)
        y = super(FlattenBlock, self).call(xf, L)
        return y



