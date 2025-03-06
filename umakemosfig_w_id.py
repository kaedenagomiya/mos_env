import toybox
from datetime import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#model_names_list = ['ljspeech',  'gradtts', 'gradseptts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
model_names_list = ['ljspeech',  'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
rename_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix deter']
color_list = ['#ff6700', '#FF0000', '#85ACFF', '#245DFE', '#468585']
hatch_list = ['','','','','']
y_min = 1
y_max = 5.4

DATA_DIR = Path('./results_evaluation')
SAVE_DIR = Path('./results_fig_mos')

#subject_id = "9567e112-c5c9-4e25-8e25-bc1bded7f2ff"
#subject_id = "81a7080b-7d46-42f8-b97a-914110ace219"
subject_id = "b33cc0b2-379c-433b-8830-f50b5d8ed46c"
#subject_id = "fa2d5dea-147c-4668-9582-be370a41cbe0"
#subject_id = "1276e5c9-8749-4542-b2a3-3530abcfb359"
#subject_id = "d14b1fd9-5fb5-4bef-bfef-fa02e24c0d4e"
#subject_id = "0136e521-d845-496d-b649-e59deb3c5a07"
#subject_id = "531d5a82-8229-43cc-96c6-48d826c1f89a"
#subject_id = "3b24700b-c374-4df7-9485-b08719beab45"
#subject_id = "5f558e4e-d68f-4a94-aaf8-ebf6df7460d9"
#subject_id = "df1d30e3-207c-45c3-9bc3-10bad7de63b7"
#subject_id = "9016a613-b237-4773-9df5-da0c9355fbc4"

fig_fileid = subject_id[0:9]
path_savefig = f'{SAVE_DIR}/mos_{fig_fileid}.png'
print(subject_id)
print(path_savefig)

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

"""
plt.figure(figsize=(10, 6))
plt.bar(mos_means.index, mos_means, yerr=mos_std, capsize=5, color='skyblue', edgecolor='black')
plt.xlabel("Model Name")
plt.ylabel("Mean Opinion Score (MOS)")
plt.title("MOS Evaluation")
plt.ylim(1, 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'{SAVE_DIR}/mos_a.png')
"""

#y_min = max(1.0, min(mos_means - mos_std) - 0.2)
#y_max = min(5.0, max(mos_means + mos_std) + 0.2)
#y_min = 1
#y_max = 5.4
#else:
#    y_min = 1
#    y_max = 5


"""
plt.figure(figsize=(8, 6))
colors = sns.color_palette(color_list, len(mos_means))
bars = plt.bar(mos_means.index, mos_means, yerr=mos_std, capsize=5, color=colors, alpha=0.7)

# ラベル設定
plt.xlabel("Model Name")
plt.ylabel("Mean Opinion Score (MOS)")
plt.title("MOS Evaluation Results with Error Bars")
plt.ylim(1, 5)

#for bar, mean in zip(bars, mos_means):
    #plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{mean:.2f}', ha='center', fontsize=12)

for bar, mean, std in zip(bars, mos_means.values, mos_std.values):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.1, f"{mean:.2f}", 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
"""

"""
# use before
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(mos_means.index, mos_means.values, yerr=mos_std.values, capsize=5, color=color_list, alpha=0.7)


for bar, mean in zip(bars, mos_means.values):
    #text_y = mean - 0.15  # バーの内部に表示（中央やや上）
    text_y = 1 + 0.2  # バーの内部に表示（中央やや上）
    text_color = "white" #if mean > (y_min + y_max) / 2 else "black"  # 明るさに応じて色を変える
    ax.text(bar.get_x() + bar.get_width() / 2, text_y, f"{mean:.2f}", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)


ax.set_ylabel("MOS Score")
ax.set_xlabel("Model Name")
ax.set_title("MOS Score")
ax.set_ylim(y_min, y_max) 
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.savefig(path_savefig)
"""


#print(type(mos_means))
#<class 'pandas.core.series.Series'>
"""
toybox.make_mosfig(
    means=mos_means,
    std=mos_std,
    color_list=color_list,
    y_min=y_min,
    y_max=y_max,
    path_savefig=path_savefig
)
"""
toybox.make_mosfig(
    means=mos_means,
    std=mos_std,
    color_list=color_list,
    hatch_list=hatch_list,
    rename_list=rename_list,
    y_min=y_min,
    y_max=y_max,
    path_savefig=path_savefig,
    ylabel="MOS Score",
    xlabel="Model Name",
    title="MOS Score"
)
