import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_DIR = Path('./results_evaluation')


for subject_dir in DATA_DIR.iterdir():
    audio_eval = pd.read_csv(subject_dir / "audio_evaluations.csv")
    start_time = datetime.strptime(audio_eval.iloc[0]['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
    end_time = datetime.strptime(audio_eval.iloc[-1]['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"{subject_dir.stem}")
    #print(f"starttime: {start_time}")
    #print(f"end_time: {end_time}")
    #print(f"apporox {elapsed_time // 60} min {elapsed_time % 60:.1f} sec (experiments_duration: {elapsed_time:.3f} sec)")
    #print()
