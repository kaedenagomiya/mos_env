import toybox
import pandas as pd
from pathlib import Path

model_names_list = ['ljspeech', 'nix_deter', 'gradtfk5tts']
DATA_DIR = Path('./results_evaluation')
subject_id = "2b005cae-245a-4d53-89fa-b02c3cce53ad"
eval_csv = DATA_DIR / subject_id / 'audio_evaluations.csv'
user_info_csv = DATA_DIR / subject_id / 'user_info.csv'

df_eval = pd.read_csv(eval_csv)
model_df=df_eval['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))
print(model_df)


df_user = pd.read_csv(user_info_csv)
exp_version = df_user['exp_version'][0]
print(f'exp_version: {exp_version}')