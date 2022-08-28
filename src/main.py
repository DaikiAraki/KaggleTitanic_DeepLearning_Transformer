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

import copy
import random
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime as dt
from src.settings.Config import Config
from src.model.InitValue import InitValue
from src.model.Handler import ModelHandler
from src.function.io_log_variable_import import write_log_header_variable_import, move_log_variable_import
from src.function.io_log_slot_import import write_log_header_slot_import
from src.function.io_value import export_value
from src.function.titanic_data_conversion import convert_X, convert_T, convert_I

"""
１：【実行方法】
　　(1) srcフォルダの存在する階層に（srcフォルダと並列に）titanic.zip（titanicで頒布されているデータセット）を配置する。
　　(2) このファイルを実行する。
２：srcフォルダの存在する階層に"data"というディレクトリが形成され、そこに結果等のデータが格納される。
３：training()が動作してから、inference()が動作する。
４：training()がモデルの構築と学習を行う。
５：inference()が提出用のデータを作成する。結果は"submission.csv"として保存される。
"""

def training(total_iteration_number):

    # 引数の条件
    assert total_iteration_number >= 1000

    # セットアップ
    cfg = Config()
    name_model = "model"
    modelHandler = ModelHandler()

    # 保存されたモデルがあれば読み込み
    initializer = InitValue()
    if len(list(cfg.path_model_data.glob("**\\*.npy"))) > 0:
        initializer.set_path(path_root=cfg.path_model_data)
        initializer.make_initValues()

    # モデル構築
    write_log_header_variable_import(name_model=name_model, path_variables=str(initializer.path_root))
    modelHandler.construct_model(initializer=initializer, name=name_model)  # model 構築, log本文書き込み
    move_log_variable_import(path_folder=cfg.path_working_dir)

    # 保存されたOptimizerのslotsがあれば読み込み
    modelHandler.model.Optimizer.set_slots_path(path=cfg.path_slots, path_log=cfg.path_slots_log)
    write_log_header_slot_import(name_model=modelHandler.model.name,
                                 path_log=cfg.path_slots_log,
                                 path_slots=cfg.path_slots)  # log本文は学習初回時に書き込まれる

    # titanicのzipデータ読み込み
    try:
        zf = zipfile.ZipFile(str(Path.cwd().parent) + "\\titanic.zip", "r")
        f_train = zf.open("train.csv", mode="r")

    except Exception as e:
        print(
            "ERROR: データが読み込めませんでした。\n" +
            "'" + str(Path.cwd().parent) + "' に Kaggle Titanic のzipデータ（titanic.zip）を配置してください。")
        raise e

    # 読み込んだデータの整形等
    pd_train = pd.read_csv(f_train, header=0, index_col=None)
    X_all = convert_X(data_rc=pd_train.values.tolist(), header=pd_train.columns.to_list())
    T_all = convert_T(data_rc=pd_train.values.tolist(), header=pd_train.columns.to_list())

    # 学習用データと汎化性能評価用データの分離
    data_num = X_all.shape[0]
    data_train_num = data_num - cfg.verify_num

    data_all_loc = [i for i in range(data_num)]
    random.shuffle(data_all_loc)

    # tensor化
    X_train = X_all[data_all_loc[:data_train_num]]
    X_verify = X_all[data_all_loc[data_train_num:]]
    T_train = T_all[data_all_loc[:data_train_num]]
    T_verify = T_all[data_all_loc[data_train_num:]]

    # 汎化性能評価が最もよかった時のモデルをbest modelとする
    loss_verify_min = None
    modelHandler_best = None

    time_start = dt.now()

    # トレーニング
    for iteration in range(total_iteration_number):

        # 学習用データと汎化誤差測定用データを用意
        miniBatch_indices = list(np.random.randint(low=0, high=data_train_num, size=cfg.miniBatch_size))

        X_train_tf = tf.convert_to_tensor(value=X_train[miniBatch_indices], dtype=tf.float32)
        T_train_tf = tf.convert_to_tensor(value=T_train[miniBatch_indices], dtype=tf.float32)
        X_verify_tf = tf.convert_to_tensor(value=X_verify, dtype=tf.float32)
        T_verify_tf = tf.convert_to_tensor(value=T_verify, dtype=tf.float32)

        loss_train = modelHandler.run_training(X=X_train_tf, Yt=T_train_tf).numpy()
        loss_verify = modelHandler.run_verification(X=X_verify_tf, Yt=T_verify_tf).numpy()

        # 誤差値を出力
        export_value(cfg.path_objective, [loss_train])
        export_value(cfg.path_objective_verification, [loss_verify])

        # 汎化誤差が最小であればモデルをキープ
        if ((loss_verify < loss_verify_min) if (loss_verify_min is not None) else True) and (iteration > 1000):
            modelHandler_best = copy.deepcopy(modelHandler)
            loss_verify_min = loss_verify

        # 経過時間表示
        if ((iteration + 1) % 1000) == 0:
            print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + str(iteration + 1) + " iterations")

    # 総経過時間表示
    time_end = dt.now()
    print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "elapsed time = " +
          f'{(time_end - time_start).seconds:,}' + "." + str((time_end - time_start).microseconds)[:3] + " [seconds]")

    print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "the training has finished")

    # best modelの保存
    modelHandler_best.model.export_variables(path_obj_dir=cfg.path_model_data)
    modelHandler_best.model.Optimizer.export_slots()
    modelHandler_best.model.Optimizer.export_iterations()
    print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "the best model has exported to " +
          "'" + str(cfg.path_model_data) + "'")

    print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "all of the processes have done. <training>")


def inference():

    # セットアップ
    cfg = Config()
    name_model = "model"
    modelHandler = ModelHandler()

    # 学習済みモデル（best model）のロード
    initializer = InitValue()
    initializer.set_path(path_root=cfg.path_model_data)
    initializer.make_initValues()

    # モデル構築
    write_log_header_variable_import(name_model=name_model, path_variables=str(initializer.path_root))
    modelHandler.construct_model(initializer=initializer, name=name_model)
    move_log_variable_import(path_folder=cfg.path_working_dir)

    # titanicのzipデータ読み込み
    try:
        zf = zipfile.ZipFile(str(Path.cwd().parent) + "\\titanic.zip", "r")
        f_test = zf.open("test.csv", mode="r")

    except Exception as e:
        print(
            "ERROR: データが読み込めませんでした。\n" +
            "'" + str(Path.cwd().parent) + "' に Kaggle Titanic のzipデータ（titanic.zip）を配置してください。")
        raise e

    # 読み込んだデータの整形等
    pd_test = pd.read_csv(f_test, header=0, index_col=None)
    X_all = convert_X(data_rc=pd_test.values.tolist(), header=pd_test.columns.to_list())
    X_index = convert_I(data_rc=pd_test.values.tolist(), header=pd_test.columns.to_list())

    # tensor化
    X_all_tf = tf.convert_to_tensor(value=X_all, dtype=tf.float32)

    # 推定値の出力
    estimations = modelHandler.run_inference_as_estimations(X=X_all_tf)  # [batch], np.ndarray

    # 推定値の保存
    df = pd.DataFrame(
        data=np.concatenate([np.expand_dims(X_index, axis=1), np.expand_dims(estimations, axis=1)], axis=1),
        index=None, columns=["PassengerId", "Survived"])
    df.to_csv(Path(str(cfg.path_working_dir) + "\\submission.csv"), index=False, header=True)

    print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "all of the processes have done. <inference>")


if __name__ == "__main__":
    training(total_iteration_number=40000)
    inference()



