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

import os
import traceback
import time
import shutil
from datetime import datetime as dt
from pathlib import Path

"""
保存しておいたtf.Variableの値（np.ndarray）をインポートした際の、読み込み結果ログ出力

【使用方法】
１．src.model.Handler.Handler の construct_model() を呼び出す前に write_log_header_...() を呼び出す
２．src.mdoel.Handler.Handler の construct_model() を呼び出した後に move_log_...() を呼び出す

※パス管理の都合上、一時的なファイルとして作成後、全てのログが書き込まれた後で、
　move_log_variable_importのなかで希望のフォルダに移動する（移動先に既にあった場合は追記される）
"""

def write_log_header_variable_import(name_model, path_variables):

    path = os.path.dirname(os.getcwd()) + "\\import_variables.log"
    try:
        with open(path, "a") as f:
            f.write("\n" + ("#" * 24) + "\n")
            f.write("  " + name_model + "    " + dt.now().strftime("%Y/%m/%d %H:%M:%S.%f") + "\n")
            f.write("  " + str(path_variables) + "\n")
            f.write(("#" * 24) + "\n")
    except:
        print(traceback.format_exc())


def save_log_variable_import(name, imported):

    path = os.path.dirname(os.getcwd()) + "\\import_variables.log"
    try:
        with open(path, "a") as f:
            f.write(name + "  \t" + (" - Imported ####" if imported else " - None ----") + "\n")
    except:
        print(traceback.format_exc())


def move_log_variable_import(path_folder):
    """
    :param path_folder: Path, target folder path
    """
    fileName = "import_variables.log"
    path_temp = Path(os.path.dirname(os.getcwd()) + "\\" + fileName)
    path_target = Path(str(path_folder) + "\\" + fileName)

    if not path_temp.exists():
        print("WARNING: \"" + str(path_temp) + "\" has not found. [move_log_variable_import()]")
        return

    if path_target.exists():  # append the file
        with open(str(path_target), "a") as f0:
            with open(str(path_temp), "r") as f1:
                row = f1.readline()
                while row != "":
                    f0.write(row)
                    row = f1.readline()
        os.remove(str(path_temp))

    else:
        for i in range(4):
            try:
                shutil.move(str(path_temp), str(path_folder))
                break
            except:
                print(traceback.format_exc())

            if i == (4 - 1):
                print("ERROR: gave up to retry move_log_variable_import()")

            time.sleep(0.500)



