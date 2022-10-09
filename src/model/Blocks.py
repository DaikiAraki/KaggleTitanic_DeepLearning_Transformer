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
    def __init__(self, inWidth, inChl, outChl, nodeQKV, actFunc, headNum=16, widthReductionRatio=2,
                 innerExpansionRatio=4, use_resPrj=True, use_firstNorm=True, use_secondNorm=True,
                 initValues=None, name="transformerBlock"):
        """
        block that mainly apply self-attention
        like a ViT or CvT or CoAtNet, construct an MLP part after the self-attention part
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param outChl: int, channel number of output
        :param nodeQKV: int, width of the query, key, value matrices of attention
        :param actFunc: activation function
        :param headNum: int, number of attention-heads
        :param widthReductionRatio: int, width reduction ratio in self-attention layer
        :param innerExpansionRatio: int, inner expansion ratio of channel (Squeeze-and-Excitation)
        :param use_resPrj: bool, whether to use residual connections
        :param use_firstNorm: bool, whether to use a normalization before self-attention
        :param use_secondNorm: bool, whether to use a normalization before mlp part
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValues
                           （attrがVariableなら、key=nameに初期化用のnumpy.ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        # << the values of 'outChl' or 'widthReductionRatio' will be invalid when 'use_resPrj' == False >>
        assert type(widthReductionRatio) in [int, type(None)]
        super(TransformerBlock, self).__init__(name=name)
        n_SaNorm = "saNorm"
        n_SelfAttention = "selfAttention"
        n_SaRes_Fc_Dw = "saResFcDw"
        n_SaHighway = "saHighway"
        n_FcNorm = "fcNorm"
        n_Fc0_Pw = "fc0Pw"
        n_ActivFc0 = "activFc0"
        n_Fc1_Pw = "fc1Pw"
        n_FcRes_Fc_Pw = "fcResFcPw"
        n_FcHighway = "fcHighway"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        if not use_resPrj:
            outChl = inChl
            widthReductionRatio = 1
        self.use_resPrj = use_resPrj
        self.outWidth = inWidth // widthReductionRatio
        self.outChl = outChl
        chl_expansion = inChl * innerExpansionRatio

        with tf.name_scope(self.name):

            """ Self-Attention Part """
            # normalization
            self.SaNorm = LayerNormLayer(shape=[inWidth, inChl], axis=[1, 2],
                                         initValues=initVal, name=n_SaNorm) if use_firstNorm else None
            # self-attention
            self.SelfAttention = SelfAttentionLayer(headNum=headNum, nodeQ=nodeQKV, nodeKV=nodeQKV, nodeO=self.outWidth,
                                                    inWidth=inWidth, inChl=inChl,
                                                    initValues=initVal, name=n_SelfAttention)
            # squeeze a width size of the residual item (because of width squeezing in the self-attention)
            self.SaRes_Fc_Dw = FcDepthwiseLayer(inWidth=inWidth, outWidth=self.outWidth, use_bias=True,
                                                initValues=initVal, name=n_SaRes_Fc_Dw) if use_resPrj else None
            # integrate the self-attention output and the residual item
            self.SaHighway = HighwayNetLayer(xShape=[self.outWidth, inChl], initValues=initVal, name=n_SaHighway)

            """ MLP Part """
            # normalization
            self.FcNorm = LayerNormLayer(shape=[self.outWidth, inChl], axis=[1, 2],
                                         initValues=initVal, name=n_FcNorm) if use_secondNorm else None
            # point-wise fc (mlp part1), Excitation in channel axis, (proposed in the "Squeeze-and-Excitatioon Network")
            self.Fc0_Pw = FcPointwiseLayer(inWidth=self.outWidth, inChl=inChl, outChl=chl_expansion, use_bias=True,
                                           initValues=initVal, name=n_Fc0_Pw)
            # activation function
            self.ActivFc0 = ActivationLayer(actFunc=actFunc, initValues=initVal, name=n_ActivFc0)
            # point-wise fc (mlp part2), Squeeze in channel axis
            self.Fc1_Pw = FcPointwiseLayer(inWidth=self.outWidth, inChl=chl_expansion, outChl=outChl, use_bias=True,
                                           initValues=initVal, name=n_Fc1_Pw)
            # squeeze a channel size of the residual (because of channel axis squeezing in the mlp)
            self.FcRes_Fc_Pw = FcPointwiseLayer(inWidth=self.outWidth, inChl=inChl, outChl=outChl, use_bias=True,
                                                initValues=initVal, name=n_FcRes_Fc_Pw) if use_resPrj else None
            # integrate the mlp output and the residual item
            self.FcHighway = HighwayNetLayer(xShape=[self.outWidth, outChl], initValues=initVal, name=n_FcHighway)

            """ register objects for calling by get_trainable_variables() and etc """
            self._register_objects({n_SaNorm: self.SaNorm,
                                    n_SelfAttention: self.SelfAttention,
                                    n_SaRes_Fc_Dw: self.SaRes_Fc_Dw,
                                    n_SaHighway: self.SaHighway,
                                    n_FcNorm: self.FcNorm,
                                    n_Fc0_Pw: self.Fc0_Pw,
                                    n_ActivFc0: self.ActivFc0,
                                    n_Fc1_Pw: self.Fc1_Pw,
                                    n_FcRes_Fc_Pw: self.FcRes_Fc_Pw,
                                    n_FcHighway: self.FcHighway})

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.bool),))
    def call(self, x, L):
        """
        :param x: [batch, inputWidth, inputChannel]
        :param L: [], whether in training
        :return y: [batch, outputWidth, outputChannel]
        """
        x_norm = self.SaNorm.call(x) if self.SaNorm is not None else x
        h_sa = self.SelfAttention.call(x_norm)
        h_saRes = self.SaRes_Fc_Dw.call(x) if self.use_resPrj else x
        h_saHighway = self.SaHighway.call(h_sa, h_saRes)
        h_fcNorm = self.FcNorm.call(h_saHighway) if self.FcNorm is not None else h_saHighway
        h_fc0 = self.ActivFc0.call(self.Fc0_Pw.call(h_fcNorm))
        h_fc1 = self.Fc1_Pw.call(h_fc0)
        h_fcRes = self.FcRes_Fc_Pw.call(h_saHighway) if self.use_resPrj else h_saHighway
        y = self.FcHighway.call(h_fc1, h_fcRes)
        return y  # [b,w,c]


class FcBlock(ModelBlock):
    # shape of x = [batch, inNode]
    # shape of y = [batch, outNode]
    def __init__(self, inNode, outNode, actFunc, use_bias=True, norm=False, pre_norm=False, dropout=False, d_rate=0.,
                 initValues=None, name="fcBlock"):
        """
        block that mainly apply the fc(= fully connected layer)
        :param inNode: int, width of input
        :param outNode: int, width of output
        :param actFunc: activation function
        :param use_bias: bool, whether to use biases in fc
        :param norm: bool, whether to use the normalization
        :param pre_norm: bool, whether to do it before fc if the normalization will be applied
        :param dropout: bool, whether to use the dropout
        :param d_rate: float, dropout ratio
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValues
                           （attrがVariableなら、key=nameに初期化用のnumpy.ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        super(FcBlock, self).__init__(name=name)
        n_Fc = "fc"
        n_Norm = "norm"
        n_Activation = "activ"
        n_Dropout = "dropout"
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.preNorm = pre_norm
        self.outNode = outNode
        shape_norm = [self.outNode] if not self.preNorm else [inNode]

        with tf.name_scope(self.name):
            self.Fc = FcLayer(inNode=inNode, outNode=outNode, use_bias=use_bias, initValues=initVal, name=n_Fc)
            self.Norm = LayerNormLayer(shape=shape_norm, axis=1, initValues=initVal, name=n_Norm) if norm else None
            self.Activation = ActivationLayer(actFunc=actFunc, initValues=initVal, name=n_Activation)
            self.Dropout = DropoutLayer(rate=d_rate, initValues=initVal, name=n_Dropout) if dropout else None
            self._register_objects({n_Fc: self.Fc,
                                    n_Norm: self.Norm,
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
        y = self.Norm.call(x) if (self.Norm is not None) and self.preNorm else x
        y = self.Fc.call(y)
        y = self.Norm.call(y) if (self.Norm is not None) and (not self.preNorm) else y
        y = self.Activation.call(y)
        y = self.Dropout.call(y, L) if (self.Dropout is not None) else y
        return y  # [b, w]


class FlattenBlock(FcBlock):
    # totalNode = width*chl
    # shape of x = [batch, width, chl]
    # shape of y = [batch, node]
    def __init__(self, inWidth, inChl, outNode, actFunc, use_bias=True, norm=False, pre_norm=False,
                 dropout=False, d_rate=0., initValues=None, name="flattenBlock"):
        """
        block that flatten input and apply fc
        :param inWidth: int, width of input
        :param inChl: int, channel number of input
        :param outNode: int, width of output (number of node in fc)
        :param actFunc: activation function
        :param use_bias: bool, whether to use biases in fc
        :param norm: bool, whether to use the normalization
        :param pre_norm: bool, whether to do it before fc if the normalization will be applied
        :param dropout: bool, whether to use the dropout
        :param d_rate: float, dropout ratio
        :param initValues: dict or None,
                           key=nameにこのブロックのinitValues
                           （attrがVariableなら、key=nameに初期化用のnumpy.ndarray）を持つ
        :param name: str, used in naming the outputing model data folder, must be unique in same hierarchy
        """
        initVal = initValues[name] if (initValues is not None) and (name in initValues.keys()) else None
        self.totalNode = inWidth * inChl  # inputNode for fc
        super(FlattenBlock, self).__init__(inNode=self.totalNode, outNode=outNode, actFunc=actFunc, use_bias=use_bias,
                                           norm=norm, pre_norm=pre_norm, dropout=dropout, d_rate=d_rate,
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



