import pandas as pd
from pathlib import Path

input_csv = Path("exp_configs/config_mos_c.csv")
output_csv = Path("exp_configs/config_mos_c_lj.csv")

if output_csv.exists():
    print(f"Error: Destination directory '{output_csv}' already exists. Aborting.")
    sys.exit(1)

original_prefix = "data/ljspeech/LJSpeech-1.1/wavs/"
new_prefix = "data/result4eval/infer4colb/lj/wav_LJ_V1/"

df = pd.read_csv(input_csv)

df["wav_path"] = df["wav_path"].apply(lambda x: x.replace(original_prefix, new_prefix) if x.startswith(original_prefix) else x)

df.to_csv(output_csv, index=False)

print(f"complete: save at {output_csv}")