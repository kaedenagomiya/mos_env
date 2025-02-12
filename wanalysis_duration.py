import os
import json
import yaml
import sys
import time
import copy
import random
from tqdm import tqdm
from pathlib import Path

import toybox

toybox.set_seed(1234)

test_ds_path = Path('configs/test_dataset.json') #Path(config['test_datalist_path'])
#RESULT_WAV_DIR_PATH = Path('data/result4eval/infer4colb/nix_deter/cpu/e500_n50/wav_LJ_V1')
#RESULT_JSON_PATH = Path('data/result4eval/infer4colb/nix_deter/cpu/e500_n50/wav_LJ_V1')
RESULT_WAV_DIR_PATH = Path('./data/ljspeech/LJSpeech-1.1/wavs')
RESULT_DURJSON_PATH = Path('./exp_configs/duration.json')
JSONL_PATH = Path('./data/result4eval/infer4colb/gradtfk5tts/cpu/e500_n50/eval4mid_LJ_V1.json')
#JSONL_PATH = Path('./data/result4eval/infer4colb/nix_deter/cpu/e500_n50/eval4mid_LJ_V1.json')

print('test_ds_path-----------------------------------------')
if test_ds_path.exists():
    print(f'Exists {str(test_ds_path)}')
    with open(test_ds_path) as j:
        test_ds_list = json.load(j)
    print(f'loaded {test_ds_path}')
else:
    print(f'No exist {test_ds_path}')
print(f"0,1:{test_ds_list[0]['name']},{test_ds_list[1]['name']}")

"""
print('durjson_path-----------------------------------------')
if RESULT_DURJSON_PATH.exists():
    print(f'Exists {str(RESULT_DURJSON_PATH)}')
    with open(RESULT_DURJSON_PATH) as j:
        durjson_list = json.load(j)
    print(f'loaded {RESULT_DURJSON_PATH}')
else:
    print(f'No exist {RESULT_DURJSON_PATH}')
"""

durjson_list = toybox.load_json(RESULT_DURJSON_PATH)
utmos_list = toybox.load_json(JSONL_PATH)

start_id = 1
max_range = 101

# duration.json has 'name', 'path', 'duration'.
# for i in range(start_id, max_range):
print(durjson_list[0])
print(durjson_list[1])

less_list = []


# 3s:7, 5s:30
for i in range(len(durjson_list)):
    if durjson_list[i]['duration'] <= 5.0:
        less_dict = {
            'name': durjson_list[i]['name'], 
            'path': durjson_list[i]['path'],
            'duration': durjson_list[i]['duration'],
        }
        less_list.append(less_dict)
    else:
        pass


print(len(less_list))
#import pdb; pdb.set_trace()
target_names = [less_list[i]['name'] for i in range(len(less_list))]

less_index = toybox.get_matching_indices(utmos_list, target_names)
#37less_index:[1, 2, 7, 8, 14, 15, 17, 22, 23, 25, 26, 27, 28, 34, 42, 45, 51, 55, 58, 61, 70, 71, 72, 73, 80, 81, 85, 91, 95]
#less_index = [1, 2, 7, 8, 14, 15, 17, 22, 23, 25, 26, 27, 28, 34, 42, 45, 51, 54, 55, 58, 61, 70, 71, 72, 73, 80, 81, 85, 91, 95]
print(f'less_index:{less_index}')
#print(less_list)
#print(len(less_list))
#print(target_names[8])
#print(less_list[8]['name'], less_list[8]['duration'])
"""
c = 0
for i in less_index:
    print(f"{c}:{utmos_list[i]['name']}")
    c += 1
"""
random_index = random.sample(less_index, 20)

c = 0
for i in random_index:
    print(f"{c}:{utmos_list[i]['name']}")
    c += 1
