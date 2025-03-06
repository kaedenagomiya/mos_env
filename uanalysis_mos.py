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


subject_id = "9567e112-c5c9-4e25-8e25-bc1bded7f2ff"
#subject_id = "81a7080b-7d46-42f8-b97a-914110ace219"
#subject_id = "b33cc0b2-379c-433b-8830-f50b5d8ed46c"
#subject_id = "fa2d5dea-147c-4668-9582-be370a41cbe0"
#subject_id = "1276e5c9-8749-4542-b2a3-3530abcfb359"
#subject_id = "d14b1fd9-5fb5-4bef-bfef-fa02e24c0d4e"
#subject_id = "0136e521-d845-496d-b649-e59deb3c5a07"
#subject_id = "531d5a82-8229-43cc-96c6-48d826c1f89a"
#subject_id = "3b24700b-c374-4df7-9485-b08719beab45"
#subject_id = "5f558e4e-d68f-4a94-aaf8-ebf6df7460d9"
#subject_id = "df1d30e3-207c-45c3-9bc3-10bad7de63b7"
#subject_id = "9016a613-b237-4773-9df5-da0c9355fbc4"

print(subject_id)
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

