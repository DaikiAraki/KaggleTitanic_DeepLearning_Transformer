author: 荒木大輝 (Araki Daiki)
e-mail: monochromaticstripes1838@gmail.com
date: 2022/08/27
========================

Kaggleの入門用課題であるtitanicについて、深層学習で予測するモデル。

別の深層強化学習でやっている課題にて作成したTransformerブロックを試す為に、そちらから一部抜粋したコードを中心に作成した。（しかし、こちらに不要なメソッド等は削除したので無駄なコードはないはずである。）

一部、Tensorflow(keras)のOptimizerソースコードを複製して改変したメソッドがあります。（その部分についてはApache License 2.0の派生成果物になります。）

性能はイマイチである（当然だが、課題に対してモデルの表現力が高すぎたのだろう）。

【追記　20221010】
SelfAttentionLayerにおいて、X→Q,K,Vへのprojectionが、width方向に行われているが、これは本来width方向に行ってはいけない。そこを誤解していたため、大きく書いて以下の修正を施した。
（Attentionの手前において、FC Layerでwidth方向に情報を混ぜてしまっていたのを改善すれば、位置に関するInductiveバイアスが増加して性能が上がる（というか本来の性能に戻る）はず）
１．Q, K, V へのprojectionをpointwiseにした。
２．multihead分を統合する際に、width方向ではなく、channel方向にconcatenate。
３．concatしたものをprojectionする時に、width方向独立な（pointwise）FCを適用。
４．softmax attentionのQKをsqrt(d_k)で割る所が、sqrtされていなかったのでそれも修正。
５．Modelクラスで作成しているInputAdaptBlockオブジェクト（中身はFC Layer）を削除した（attention以外によるwidth方向の情報の混ぜ合わせを回避するため）。
６．TransformerBlockクラスでdownsamplingが行われなくなるので、代わりにBlock末尾にDownsampling用のConvLayerを追加した。stride=2の通常のConvolution。
