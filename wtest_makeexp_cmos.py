"""
To check wav_duration and To output duration.json
>>> python3 wtimecheck.py
To make csv for list up wav_name
>>> python3 wanalysis_duration.py
And do this
>>> python3 wmakeexp.py
"""

import os
import csv
import json
import yaml
import sys
import time
import copy
import itertools
import random
from tqdm import tqdm
from pathlib import Path
from itertools import permutations
import pandas as pd
import toybox

test_ds_path = Path('configs/test_dataset.json')
BASE_WAV_DIR_PATH = Path('./data/result4eval/infer4colb')
EXP_CONFIG_DIR_PATH = Path('./exp_configs')

LJ_WAV_DIR_PATH = Path('./data/ljspeech/LJSpeech-1.1/wavs')
TFK5_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'gradtfk5tts' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
NIXD_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'nix_deter' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
WAV_DIR_LIST = [LJ_WAV_DIR_PATH, TFK5_WAV_DIR_PATH, NIXD_WAV_DIR_PATH]

CONFIG_A_PATH = EXP_CONFIG_DIR_PATH / 'config_wavs_a.csv'
CONFIG_B_PATH = EXP_CONFIG_DIR_PATH / 'config_wavs_b.csv'
CONFIG_C_PATH = EXP_CONFIG_DIR_PATH / 'config_wavs_c.csv'
CONFIG_PATH_LIST = [CONFIG_A_PATH]
#CONFIG_PATH_LIST = [CONFIG_A_PATH, CONFIG_B_PATH, CONFIG_C_PATH]

CONFIG_CMOS_A_PATH = EXP_CONFIG_DIR_PATH / 'config_cmos_a.csv'
CONFIG_CMOS_B_PATH = EXP_CONFIG_DIR_PATH / 'config_cmos_b.csv'
CONFIG_CMOS_C_PATH = EXP_CONFIG_DIR_PATH / 'config_cmos_c.csv'
CONFIG_CMOS_LIST = [CONFIG_CMOS_A_PATH, CONFIG_CMOS_B_PATH, CONFIG_CMOS_C_PATH]

pairs = []

for i in range(len(CONFIG_PATH_LIST)):
    
    config_path = CONFIG_PATH_LIST[i]

    # get wav_name
    df = pd.read_csv(config_path)
    wav_name_list = df['wav_name'].tolist()

    # get path about each wav_name
    wav_path_list = {filename: [] for filename in wav_name_list}

    print(wav_path_list)

    for wav_dir_path in WAV_DIR_LIST:
        for wav_name in wav_name_list:
            wav_path = wav_dir_path / f"{wav_name}.wav"
            if wav_path.exists():
                wav_path_list[wav_name].append(str(wav_path))

    # get pair
    for filename, paths in wav_path_list.items():
        for a_wav, b_wav in itertools.permutations(paths, 2):  # consider the order
            pairs.append([a_wav, b_wav])

    
    # convert to DataFrame
    comparison_df = pd.DataFrame(pairs, columns=["wav1", "wav2"])
    print(comparison_df)
    # save to csv
    comparison_df.to_csv(CONFIG_CMOS_LIST[i], index=False)



"""
check:
csv_path = './exp_configs/config_wavs_b.csv'
csv_data = pd.read_csv(csv_path)
csv_list = csv_data.name.values.tolist()
mos_wav = pd.read_csv('./exp_configs/CONFIG_CMOS_b.csv')

>>> for i in csv_list:
...     for j in mos_wav['wav_path']:
...             if re.search(re.escape(i), j):
...                     print(f'match:{i}: {j}')

match:LJ020-0038: data/ljspeech/LJSpeech-1.1/wavs/LJ020-0038.wav
match:LJ020-0038: data/result4eval/infercolb/gradtfk5tts/cpu/e500_n50/wav_LJ_V1/LJ020-0038.wav
match:LJ020-0038: data/result4eval/infercolb/nix_deter/cpu/e500_n50/wav_LJ_V1/LJ020-0038.wav
match:LJ011-0016: data/ljspeech/LJSpeech-1.1/wavs/LJ011-0016.wav
match:LJ011-0016: data/result4eval/infercolb/nix_deter/cpu/e500_n50/wav_LJ_V1/LJ011-0016.wav
match:LJ011-0016: data/result4eval/infercolb/gradtfk5tts/cpu/e500_n50/wav_LJ_V1/LJ011-0016.wav
...
"""