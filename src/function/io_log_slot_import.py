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

import traceback
from datetime import datetime as dt

"""
保存しておいた Optimizer の slots (moment類の保管庫) の値（np.ndarray）をインポートした際の、読み込み結果ログ出力

【使用方法】
１．src.model.Handler.Handler の run_training() を呼び出す前に write_log_header_...() を呼び出す
"""


def write_log_header_slot_import(name_model, path_log, path_slots):
    try:
        with open(path_log, "a") as f:
            f.write("\n" + ("#" * 24) + "\n")
            f.write("  " + name_model + "    " + dt.now().strftime("%Y/%m/%d %H:%M:%S.%f") + "\n")
            f.write("  " + str(path_slots) + "\n")
            f.write(("#" * 24) + "\n")
    except:
        print(traceback.format_exc())


def save_log_slot_import(var_key, slot_name, imported, path):
    try:
        with open(path, "a") as f:
            f.write(var_key + " : " + slot_name + "  \t" + (" - Imported ####" if imported else " - None ----") + "\n")
    except:
        print(traceback.format_exc())



