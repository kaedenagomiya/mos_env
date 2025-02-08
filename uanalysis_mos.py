import toybox
from datetime import datetime
import pandas as pd
from pathlib import Path

#model_names_list = ['ljspeech',  'gradtts', 'gradseptts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
model_names_list = ['ljspeech',  'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
DATA_DIR = Path('./results_evaluation')
#subject_id = "2b005cae-245a-4d53-89fa-b02c3cce53ad"
subject_id = "924af422-dc2c-4880-8c39-e512c83e8448"
eval_csv = DATA_DIR / subject_id / 'audio_evaluations.csv'
user_info_csv = DATA_DIR / subject_id / 'user_info.csv'

df_eval = pd.read_csv(eval_csv)
#model_df=df_eval['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))
#print(model_df)
start_time = datetime.fromisoformat(df_eval['timestamp'][0])
end_time = datetime.fromisoformat(df_eval['timestamp'][len(df_eval)-1])
time_diff =  end_time - start_time
print(f'time_diff: {time_diff}')
mos_means = df_eval.groupby("model_name")["evaluation"].mean()
print(mos_means)

#df_user = pd.read_csv(user_info_csv)
#exp_version = df_user['exp_version'][0]
#print(f'exp_version: {exp_version}')
