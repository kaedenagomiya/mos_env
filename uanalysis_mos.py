import toybox
from datetime import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#model_names_list = ['ljspeech',  'gradtts', 'gradseptts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
model_names_list = ['ljspeech',  'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
DATA_DIR = Path('./results_evaluation')
SAVE_DIR = Path('./results_fig_mos')
#subject_id = "2b005cae-245a-4d53-89fa-b02c3cce53ad"
#subject_id = "924af422-dc2c-4880-8c39-e512c83e8448"

#subject_id = "5f558e4e-d68f-4a94-aaf8-ebf6df7460d9"
#subject_id = "0136e521-d845-496d-b649-e59deb3c5a07"
#subject_id = "d14b1fd9-5fb5-4bef-bfef-fa02e24c0d4e"
subject_id = "df1d30e3-207c-45c3-9bc3-10bad7de63b7"
# [tmp] = "588291ca-85cc-4301-842c-015334541856"
#subject_id = "81a7080b-7d46-42f8-b97a-914110ace219" #[x]

eval_csv = DATA_DIR / subject_id / 'audio_evaluations.csv'
user_info_csv = DATA_DIR / subject_id / 'user_info.csv'

df_eval = pd.read_csv(eval_csv)
#model_df=df_eval['wav_path'].apply(lambda path: toybox.extract_model_name(path, model_names_list))
#print(model_df)
print(f"{len(df_eval)}")
start_time = datetime.fromisoformat(df_eval['timestamp'][0])
end_time = datetime.fromisoformat(df_eval['timestamp'][len(df_eval)-1])
time_diff =  end_time - start_time
print(f'time_diff: {time_diff}')
mos_means = df_eval.groupby("model_name")["evaluation"].mean()
mos_means = mos_means.reindex(model_names_list)
mos_std = df_eval.groupby("model_name")["evaluation"].std()
mos_std = mos_std.reindex(model_names_list)
print(f"mos_mean: \n {mos_means}")
#print(f"mos_std: \n {mos_std}")

#df_user = pd.read_csv(user_info_csv)
#exp_version = df_user['exp_version'][0]
#print(f'exp_version: {exp_version}')

