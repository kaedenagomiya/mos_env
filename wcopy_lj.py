import json
import sys
import shutil
from pathlib import Path
import toybox

jsonl_path = "configs/test_dataset.json"
source_base_dir = Path("data/ljspeech/LJSpeech-1.1/wavs")
destination_dir = Path("data/result4eval/infer4colb/lj/wav_LJ_V1")

test_info = toybox.load_json(jsonl_path)[0]

if destination_dir.exists():
    print(f"Error: Destination directory '{destination_dir}' already exists. Aborting.")
    sys.exit(1)

destination_dir.mkdir(parents=True)

for i in range(len(test_info)):
    wav_path = Path(test_info[i]["wav_path"])
    wav_name = wav_path.name

    source_path = source_base_dir / wav_name
    destination_path = destination_dir / wav_name
    #print(destination_path)
    if source_path.exists():
        shutil.copy(source_path, destination_path)
        print(f"Copied: {source_path} -> {destination_path}")
    else:
        print(f"Warning: {source_path} not found, skipping.")
