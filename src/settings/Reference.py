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

"""
設定値
変更した場合、モデルのリセットが必要になるもの（モデル構成など）
"""

class Reference:

    def __init__(self):
        pass

    """ input & output """
    input_node = 20
    output_node = 2

    """ optimizer """
    opt_lr = 1.e-4
    opt_momentum = 0.900
    opt_rmsprop = 0.999
    opt_epsilon = 1.e-8

    """ model structure """
    inputAdaptBlock_output_node = 128
    inputAdaptBlock_norm = True
    inputAdaptBlock_dropout = False
    inputAdaptBlock_dropout_rate = 0.00

    transformerBlocks_num = 2
    transformerBlocks_outChl = [2, 4]
    transformerBlocks_nodeQKV = [16, 16]
    transformerBlocks_head_num = [4, 4]
    transformerBlocks_width_reduction_ratio = [2, 2]
    transformerBlocks_inner_expansion_ratio = [4, 4]
    transformerBlocks_use_res_prj = [False, False]
    transformerBlocks_use_first_norm = [True, True]
    transformerBlocks_use_second_norm = [True, True]

    flattenBlock_outNode = 32
    flattenBlock_norm = True
    flattenBlock_dropout = True
    flattenBlock_dropout_rate = 0.20



