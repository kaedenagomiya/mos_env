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

#model_names_list = ['ljspeech', 'gradtts', 'gradseptts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
model_names_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']

CONFIG_MOS_A_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_a.csv'
CONFIG_MOS_B_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_b.csv'
CONFIG_MOS_C_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_c.csv'

print('MOS_lists-----------------------------------------')
if CONFIG_MOS_A_PATH.exists() or \
    CONFIG_MOS_B_PATH.exists() or \
    CONFIG_MOS_C_PATH.exists():
    print(f'Already Exists wav_path.csv for MOS')
    #raise Exception('Cannot overwrite an already existing file.')
else:
    print(f'No exist {CONFIG_MOS_A_PATH}')
    print(f'No exist {CONFIG_MOS_B_PATH}')
    print(f'No exist {CONFIG_MOS_C_PATH}')


df_a = pd.read_csv(CONFIG_MOS_A_PATH)
df_b = pd.read_csv(CONFIG_MOS_B_PATH)
df_c = pd.read_csv(CONFIG_MOS_C_PATH)
df_a['model_name'] = df_a['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))
df_b['model_name'] = df_b['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))
df_c['model_name'] = df_c['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))

print(df_a)
