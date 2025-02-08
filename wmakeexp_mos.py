"""
To check wav_duration and To output duration.json
>>> python3 wtimecheck.py
To make csv for list up wav_name
>>> python3 wanalysis_duration.py
And do this
>>> python3 wmakeexp.py
"""

import os
import json
import yaml
import sys
import time
import copy
import random
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import toybox

test_ds_path = Path('configs/test_dataset.json')
BASE_WAV_DIR_PATH = Path('./data/result4eval/infer4colb')
EXP_CONFIG_DIR_PATH = Path('./exp_configs')

LJ_WAV_DIR_PATH = Path('./data/ljspeech/LJSpeech-1.1/wavs')
GT_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'gradtts' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
#SGT_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'gradseptts' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
TFK_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'gradtfktts' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
TFK5_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'gradtfk5tts' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
NIXD_WAV_DIR_PATH = BASE_WAV_DIR_PATH / 'nix_deter' / 'cpu' / 'e500_n50' / 'wav_LJ_V1'
WAV_DIR_LIST = [LJ_WAV_DIR_PATH, GT_WAV_DIR_PATH, TFK_WAV_DIR_PATH, TFK5_WAV_DIR_PATH, NIXD_WAV_DIR_PATH]
#WAV_DIR_LIST = [LJ_WAV_DIR_PATH, GT_WAV_DIR_PATH, SGT_WAV_DIR_PATH,TFK_WAV_DIR_PATH, TFK5_WAV_DIR_PATH, NIXD_WAV_DIR_PATH]

CONFIG_A_PATH = EXP_CONFIG_DIR_PATH / 'config_wavs_a.csv'
CONFIG_B_PATH = EXP_CONFIG_DIR_PATH / 'config_wavs_b.csv'
CONFIG_C_PATH = EXP_CONFIG_DIR_PATH / 'config_wavs_c.csv'

CONFIG_MOS_A_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_a.csv'
CONFIG_MOS_B_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_b.csv'
CONFIG_MOS_C_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_c.csv'


print('test_ds_path-----------------------------------------')
if test_ds_path.exists():
    print(f'Exists {str(test_ds_path)}')
    with open(test_ds_path) as j:
        test_ds_list = json.load(j)
    print(f'loaded {test_ds_path}')
else:
    print(f'No exist {test_ds_path}')

print('wav for exp-----------------------------------------')
if LJ_WAV_DIR_PATH.exists() and \
    TFK5_WAV_DIR_PATH.exists() and \
    NIXD_WAV_DIR_PATH.exists():
    print(f'Exists all wavfile for exp.')
else:
    print(f'No exist all wavfile for exp.')


print('MOS_lists-----------------------------------------')
if CONFIG_MOS_A_PATH.exists() or \
    CONFIG_MOS_B_PATH.exists() or \
    CONFIG_MOS_C_PATH.exists():
    print(f'Already Exists wav_path.csv for MOS')
    raise Exception('Cannot overwrite an already existing file.')
else:
    print(f'Make {CONFIG_MOS_A_PATH}')
    print(f'Make {CONFIG_MOS_B_PATH}')
    print(f'Make {CONFIG_MOS_C_PATH}')


a = pd.read_csv(CONFIG_A_PATH)
b = pd.read_csv(CONFIG_B_PATH)
c = pd.read_csv(CONFIG_C_PATH)
a_list = a.wav_name.values.tolist()
b_list = b.wav_name.values.tolist()
c_list = c.wav_name.values.tolist()

wav_path = []
wav_path_list_a = []
wav_path_list_b = []
wav_path_list_c = []
# make wav_path_list
for dir_path in WAV_DIR_LIST:
    for wav_name_a in a_list:
        wav_path_a = str(dir_path / f'{wav_name_a}.wav')
        wav_path_list_a.append(wav_path_a)

    for wav_name_b in b_list:
        wav_path_b = str(dir_path / f'{wav_name_b}.wav')
        wav_path_list_b.append(wav_path_b)

    for wav_name_c in c_list:
        wav_path_c = str(dir_path / f'{wav_name_c}.wav')
        wav_path_list_c.append(wav_path_c)


print('=====================================')
print(wav_path_list_a)
print('=====================================')
#print(wav_path_list_b)
#print('=====================================')
#print(wav_path_list_c)
#print('=====================================')
print(len(wav_path_list_a))
#print(len(wav_path_list_b))
#print(len(wav_path_list_c))

# shuffle
shuffle_wav_path_a = random.sample(wav_path_list_a, len(wav_path_list_a))
shuffle_wav_path_b = random.sample(wav_path_list_b, len(wav_path_list_b))
shuffle_wav_path_c = random.sample(wav_path_list_c, len(wav_path_list_c))
print('=====================================')

# output csv
df_a = pd.DataFrame(shuffle_wav_path_a, columns=['wav_path'])
df_b = pd.DataFrame(shuffle_wav_path_b, columns=['wav_path'])
df_c = pd.DataFrame(shuffle_wav_path_c, columns=['wav_path'])

df_a.to_csv(CONFIG_MOS_A_PATH, index=False)
df_b.to_csv(CONFIG_MOS_B_PATH, index=False)
df_c.to_csv(CONFIG_MOS_C_PATH, index=False)
print('fin')

"""
check:
csv_path = './exp_configs/config_wavs_b.csv'
csv_data = pd.read_csv(csv_path)
csv_list = csv_data.name.values.tolist()
mos_wav = pd.read_csv('./exp_configs/config_MOS_b.csv')

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