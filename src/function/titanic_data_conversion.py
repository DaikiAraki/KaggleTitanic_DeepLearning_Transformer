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

def convert_X(data_rc, header):
    """
    モデルに入力するデータを作成
    :param data_rc: 2d list, [row, column] = [batch, width]
    :param header: 1d list, corresponds to data_rc's column
    :return: 2d np.ndarray, [batch, width]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    # make a list of 2d np.ndarray
    dataList = [
        convert_Pclass(data=data[header.index("Pclass")]),
        convert_Name(data=data[header.index("Name")]),
        convert_Sex(data=data[header.index("Sex")]),
        convert_Age(data=data[header.index("Age")]),
        convert_SibSp(data=data[header.index("SibSp")]),
        convert_Parch(data=data[header.index("Parch")]),
        convert_Fare(data=data[header.index("Fare")]),
        convert_Cabin(data=data[header.index("Cabin")]),
        convert_Embarked(data=data[header.index("Embarked")])]

    X = np.concatenate(dataList, axis=0).T
    return X  # np.ndarray, [batch, width]

def convert_T(data_rc, header):
    """
    正解データを抽出
    :param data_rc: 2d list, [row, column]
    :param header: 1d list, corresponds to data_rc's column
    :return: 2d np.ndarray, [batch, width]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    T = convert_Survived(data=data[header.index("Survived")]).T
    return T  # np.ndarray, [batch, width]

def convert_I(data_rc, header):
    """
    データIDを抽出
    :param data_rc: 2d list, [row, column]
    :param header: 1d list, corresponds to data_rc's column
    :return: 1d np.ndarray, [batch]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    I = np.array(data[header.index("PassengerId")], dtype=np.int32)
    return I  # np.ndarray, [batch]

def convert_Survived(data):  # channel=2
    """
    生存したかどうか（教師値）を値化
    （0: 死，1: 生）を [死亡, 生存] の形式で、当てはまる方で 1、それ以外は 0 を取るnp.ndarrayにする
    :param data: Survived, 1d list
    :return: [dead, alive]
    """
    res = []
    for i in range(len(data)):
        if data[i] == 0:
            res.append([1, 0])
        elif data[i] == 1:
            res.append([0, 1])
        else:
            raise Exception(
                "ERROR: detect missing data in [model.modelFunction.dataConversion.convert_Survived]\n" +
                "argument 'data'[" + str(i) + "] = " + str(data[i]))

    return np.array(res, dtype=np.float32).T  # [batch, 2]

def convert_Pclass(data):  # channel=1
    """
    乗客のチケットクラスを数値化
    「良い」方が値が大きくなるように
    :param data: Pclass, 1d list
    :return: [pclass]
    """
    res = []
    for i in range(len(data)):
        if data[i] == 1:
            res.append([3])
        elif data[i] == 2:
            res.append([2])
        elif data[i] == 3:
            res.append([1])
        else:
            raise Exception(
                "ERROR: encountered invalid value in [model.modelFunction.dataConversion.convert_Pclass()]\n" +
                "argument 'data' = " + str(data))

    return  np.array(res, dtype=np.float32).T  # [batch, 1]

def convert_Name(data):  # channel=2
    """
    記載氏名から敬称が有るか無いかをクラス化
    :param data: Name
    :return: [normal, ranker]
    """
    res = []
    for i in range(len(data)):
        if data[i] in [np.nan]:
            res.append([0, 1])
        elif (("Mr." in data[i]) or ("Miss." in data[i]) or ("Ms." in data[i]) or
                ("Mrs." in data[i]) or ("Mme." in data[i]) or ("Mlle." in data[i])):
            res.append([1, 0])
        else:
            res.append([0, 1])
    return np.array(res, dtype=np.float32).T  # [batch, 2]

def convert_Sex(data):  # channel=2
    """
    性別をクラス化
    :param data: Sex
    :return: [male, female]
    """
    res = []
    for i in range(len(data)):
        if data[i] == "male":
            res.append([1, 0])
        elif data[i] == "female":
            res.append([0, 1])
        else:
            res.append([0, 0])
    return np.array(res, dtype=np.float32).T  # [batch, 2]

def convert_Age(data):  # channel=1
    """
    年齢
    :param data: Age
    :return: [age]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_SibSp(data):  # channel=1
    """
    一緒に乗船している siblings or spouses の数
    :param data: SibSp
    :return: [sibsp]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_Parch(data):  # channel=1
    """
    一緒に乗船している parents or children の数
    :param data: Parch
    :return: [parch]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_Fare(data):  # channel=1
    """
    運賃
    :param data: Fare
    :return: [fare]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_Cabin(data):  # channel=8
    """
    客室番号の英数字をクラス化
    :param data: Cabin
    :return: [A, B, C, D, E, F, G, ELSE]
    """
    res = []
    for i in range(len(data)):
        if data[i] in ["", np.nan]:
            res.append([0, 0, 0, 0, 0, 0, 0, 0])
        else:
            r = [0, 0, 0, 0, 0, 0, 0, 0]
            if "A" in data[i]:
                r[0] = 1
            if "B" in data[i]:
                r[1] = 1
            if "C" in data[i]:
                r[2] = 1
            if "D" in data[i]:
                r[3] = 1
            if "E" in data[i]:
                r[4] = 1
            if "F" in data[i]:
                r[5] = 1
            if "G" in data[i]:
                r[6] = 1
            if sum(r) == 0:
                r[7] = 1
            res.append(r)
    return np.array(res, dtype=np.float32).T  # [batch, 8]

def convert_Embarked(data):  # channel=3
    """
    乗船港をクラス化
    :param data: Embarked
    :return: [C, Q, S]
    """
    res = []
    for i in range(len(data)):
        r = [0, 0, 0]
        if data[i] in [np.nan]:
            pass
        elif "C" in data[i]:
            r[0] = 1
        elif "Q" in data[i]:
            r[1] = 1
        elif "S" in data[i]:
            r[2] = 1
        res.append(r)
    return np.array(res, dtype=np.float32).T  # [batch, 3]



