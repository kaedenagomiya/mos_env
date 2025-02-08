"""
python3 wtimecheck.py 
then output duration.json
"""
import os
import json
import yaml
import sys
import time
import copy
from tqdm import tqdm
from pathlib import Path
import torchaudio

import toybox


test_ds_path = Path('configs/test_dataset.json') #Path(config['test_datalist_path'])
#RESULT_WAV_DIR_PATH = Path('data/result4eval/infer4colb/nix_deter/cpu/e500_n50/wav_LJ_V1')
#RESULT_JSON_PATH = Path('data/result4eval/infer4colb/nix_deter/cpu/e500_n50/wav_LJ_V1')
RESULT_WAV_DIR_PATH = Path('./data/ljspeech/LJSpeech-1.1/wavs')
RESULT_JSON_PATH = Path('./configs/duration.json')


print('test_ds_path-----------------------------------------')
if test_ds_path.exists():
    print(f'Exists {str(test_ds_path)}')
    with open(test_ds_path) as j:
        test_ds_list = json.load(j)
    print(f'loaded {test_ds_path}')
else:
    print(f'No exist {test_ds_path}')

infer_data_num: int = 101 #101 #len(test_ds_list) is 200
save_list = []

for i in tqdm(range(infer_data_num)):
    test_ds_filename = test_ds_list[i]['name']
    #mel_npy_path = RESULT_MEL_DIR_PATH / f"{test_ds_filename}.npy"
    synth_wav_path = RESULT_WAV_DIR_PATH / f"{test_ds_filename}.wav"
    print(f'test_ds_index_{i}: {test_ds_filename}')
    wav, samplerate = torchaudio.load(synth_wav_path)
    # metadata = torchaudio.info()
    length_wav = len(wav[0])
    #print(length_wav, samplerate)
    dur_time = length_wav / samplerate
    print(f'{synth_wav_path}: {dur_time}')
    save_dict = {
        'name': test_ds_filename,
        'path': str(synth_wav_path),
        'duration': dur_time
    }
    save_list.append(save_dict)



if RESULT_JSON_PATH.exists() == False:
    with open(RESULT_JSON_PATH, 'w') as f:
        for entry in save_list:
            f.write(json.dumps(entry) + '\n')
    print(f'Make {RESULT_JSON_PATH}')
else:
    print(f'Already Exists {RESULT_JSON_PATH}')