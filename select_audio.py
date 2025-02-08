import json
import datetime
import uuid
from pathlib import Path
import pandas as pd

import toybox

max_num4play = 10
config_wav_path = "configs/test_dataset.json"
test_config_list = toybox.load_json(config_wav_path)[0]
#{'name': 'LJ031-0171', 'wav_path': 'data/ljspeech/LJSpeech-1.1/wavs/LJ031-0171.wav', 'text': "", 'phonemes': []}
audioinfo_list = [test_config_list[i]['name'] for i in range(len(test_config_list))]
audioinfo_list = audioinfo_list[1:max_num4play]
audiopath_list = [f"./data/ljspeech/LJSpeech-1.1/wavs/{wav_name}.wav" for wav_name in audioinfo_list]
length_evalset = len(audiopath_list)


# 時間数が短いものを選定10秒未満
# ファイルパスを取り出す
# ランダムに選択(各モデルで同じaudio)
# 選んだファイルを更にランダム()
# ファイルに書き出し