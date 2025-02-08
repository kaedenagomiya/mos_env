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

#model_names_list = ['ljspeech', 'gradtts', 'gradseptts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
model_names_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']

EXP_CONFIG_DIR_PATH = Path('./exp_configs')
CONFIG_MOS_A_PATH = EXP_CONFIG_DIR_PATH / 'config_mos_a.csv'
RESULT_DIR = Path('./results_evaluation')
subject_id = "2b005cae-245a-4d53-89fa-b02c3cce53ad"
RESULT_CSV_PATH = RESULT_DIR / subject_id / 'audio_evaluations.csv'

df = pd.read_csv(RESULT_CSV_PATH)

df['model_name'] = df['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))

mos_means = df.groupby("model_name")["evaluation"].mean()

print(mos_means)