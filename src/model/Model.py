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

from src.settings.Reference import Reference as Ref
from src.model.ModelCore import ModelCore
from src.model.Blocks import *
from src.model.MinimizingObjectives import ObjectiveProbability
from src.model.OptimizerAdaBelief import AdaBelief
from src.function.model_functions import mish, softmax

"""
モデル本体
trainingさせるなどの機能はHandlerに持たせている（このModelクラスは単にBlocksやLayersを統合したもの）

### CAUTION ###
the all Variables names (tf.Variable._shared_name) must be unique in the model
the all Variables names must not be named like "xxx_12". do not use "_"+number at the end of the name.
"""

class Model(ModelCore):

    def __init__(self, initValues=None, name="model"):
        """
        :param initValues: output of the InitValue.get_initValues() or None
        :param name: str, used in naming the outputing model data folder
        """
        super(Model, self).__init__(name=name)
        self.out_node = Ref.output_node

        """ Object Names: all name is must be unique """
        n_InputAdaptBlock = "inputAdaptBlock"
        n_ExpandDimLayer = "expandDimLayer"
        n_TransformerBlocks = ["transformerBlock" + f'{i:02d}' for i in range(Ref.transformerBlocks_num)]
        n_FlattenBlock = "flattenBlock"
        n_OutputBlock = "outputBlock"

        """ Objects """
        self._objects = {}  # Model classが直接持っているBlocksやLayers（のdict）
        self._trainable_variables = []  # 学習可能なVariables, サブクラス内も含めた全てをflat listとして持つ
        self._variables = {}  # Model class以下の全てのvariablesを階層状dictで持つ

        """ model """
        # 入力を整形する意味合いも兼ねたFC
        self.InputAdaptBlock = FcBlock(
            inNode=Ref.input_node, outNode=Ref.inputAdaptBlock_output_node, actFunc=mish, use_bias=True,
            norm=Ref.inputAdaptBlock_norm, pre_norm=False,
            dropout=Ref.inputAdaptBlock_dropout, d_rate=Ref.inputAdaptBlock_dropout_rate,
            initValues=initValues, name=n_InputAdaptBlock)
        # FC出力をTransformer Blockに入れる為にチャネル次元を追加する
        self.ExpandDimLayer = ReshapingLayer(
            shape=[self.InputAdaptBlock.outNode, 1], initValues=initValues, name=n_ExpandDimLayer)
        # Transformer Blockをいくつか重ねる
        self.TransformerBlocks = []
        for i in range(Ref.transformerBlocks_num):
            self.TransformerBlocks.append(
                TransformerBlock(
                    inWidth=self.InputAdaptBlock.outNode if (i == 0) else self.TransformerBlocks[-1].outWidth,
                    inChl=1 if (i == 0) else self.TransformerBlocks[-1].outChl,
                    outChl=Ref.transformerBlocks_outChl[i], nodeQKV=Ref.transformerBlocks_nodeQKV[i],
                    actFunc=mish, headNum=Ref.transformerBlocks_head_num[i],
                    widthReductionRatio=Ref.transformerBlocks_width_reduction_ratio[i],
                    innerExpansionRatio=Ref.transformerBlocks_inner_expansion_ratio[i],
                    use_resPrj=Ref.transformerBlocks_use_res_prj[i],
                    use_firstNorm=Ref.transformerBlocks_use_first_norm[i],
                    use_secondNorm=Ref.transformerBlocks_use_second_norm[i],
                    initValues=initValues, name=n_TransformerBlocks[i]))
        # Transformer出力をflattenし、fcを適用
        self.FlattenBlock = FlattenBlock(
            inWidth=self.TransformerBlocks[-1].outWidth, inChl=self.TransformerBlocks[-1].outChl,
            outNode=Ref.flattenBlock_outNode, actFunc=mish, use_bias=True, norm=Ref.flattenBlock_norm, pre_norm=True,
            dropout=Ref.flattenBlock_dropout, d_rate=Ref.flattenBlock_dropout_rate,
            initValues=initValues, name=n_FlattenBlock)
        # fc出力として最終的な出力を得る, 確率値（0 <= output <= 1）である
        self.OutputBlock = FcBlock(
            inNode=self.FlattenBlock.outNode, outNode=Ref.output_node, actFunc=softmax, use_bias=True,
            norm=False, pre_norm=False, dropout=False, d_rate=False, initValues=initValues, name=n_OutputBlock)

        """ Objectives """
        self.ObjectiveProbability = ObjectiveProbability()  # 誤差計算クラスのinstance作成

        """ Optimizer """
        self.Optimizer = AdaBelief(learning_rate=Ref.opt_lr, momentum=Ref.opt_momentum, rmsprop=Ref.opt_rmsprop,
                                   epsilon=Ref.opt_epsilon, name="OptimizerAdaBelief")

        """ register objects """
        self._register_objects(
            obj_dict={**{n_InputAdaptBlock: self.InputAdaptBlock,
                         n_ExpandDimLayer: self.ExpandDimLayer},
                      **dict((n_TransformerBlocks[blk], self.TransformerBlocks[blk])
                             for blk in range(len(self.TransformerBlocks))),
                      **{n_FlattenBlock: self.FlattenBlock,
                         n_OutputBlock: self.OutputBlock}})


    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[], dtype=tf.bool),))
    def call(self, X, L):
        """
        :param X: [batch, width], input
        :param L: tf.bool, whether in training
        :return Y: [batch, outputWidth], output
        """
        h_inputAdaptBlock = self.InputAdaptBlock.call(x=X, L=L)  # [b,w]
        h_expandDimLayer = self.ExpandDimLayer.call(x=h_inputAdaptBlock)  # [b,w,1]
        h_transformerBlocks = []
        for blk in range(len(self.TransformerBlocks)):
            x = h_expandDimLayer if (blk == 0) else h_transformerBlocks[-1]
            h_transformerBlocks.append(self.TransformerBlocks[blk].call(x=x, L=L))    # [b,w,c]
        h_flattenBlock = self.FlattenBlock.call(x=h_transformerBlocks[-1], L=L)  # [b,w]
        Y = self.OutputBlock.call(x=h_flattenBlock, L=L)  # [b,w]
        return Y  # [b,w]


    def get_trainable_variables(self):
        """
        training時に学習可能Variablesを得る為のmethod
        :return: 学習可能な全Variables（1d list）
        """
        return self._trainable_variables


    def get_objects(self):
        """
        このクラスが直接持つモデルの一部分を全て取得
        :return: 全てのModelCoreのサブクラスのインスタンス
        """
        return self._objects



