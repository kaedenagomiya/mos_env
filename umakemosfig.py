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
#subject_id = "df1d30e3-207c-45c3-9bc3-10bad7de63b7"
# [tmp] = "588291ca-85cc-4301-842c-015334541856"
#subject_id = "81a7080b-7d46-42f8-b97a-914110ace219" #[x]
y_min = 1
y_max = 5.4
color_list = ['#ff6700', '#FF0000', '#85ACFF', '#245DFE', '#468585']

for subject_dir in DATA_DIR.iterdir():
    if not subject_dir.is_dir():
        continue

    subject_id = subject_dir.name
    eval_csv = subject_dir / 'audio_evaluations.csv'

    if not eval_csv.exists():
        print(f"Skipping {subject_id}: No evaluation file found.")
        continue

    df_eval = pd.read_csv(eval_csv)

    if len(df_eval) < 99:
        print(f"Skipping {subject_id}: Not enough data ({len(df_eval)} rows).")
        continue 

    mos_means = df_eval.groupby("model_name")["evaluation"].mean()
    mos_means = mos_means.reindex(model_names_list)
    mos_std = df_eval.groupby("model_name")["evaluation"].std()
    mos_std = mos_std.reindex(model_names_list) 

    # for figure
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(mos_means.index, mos_means.values, yerr=mos_std.values, capsize=5, color=color_list, alpha=0.7)

    for bar, mean in zip(bars, mos_means.values):
        #text_y = mean - 0.15  # バーの内部に表示（中央やや上）
        #text_y = max(y_min + 0.2, mean - 0.2)
        text_y = 1 + 0.2  # バーの内部に表示（中央やや上）
        text_color = "white" #if mean > (y_min + y_max) / 2 else "black"  # 明るさに応じて色を変える
        ax.text(bar.get_x() + bar.get_width() / 2, text_y, f"{mean:.2f}", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)

    ax.set_ylabel("MOS Score")
    ax.set_xlabel("Model Name")
    #ax.set_title("MOS Score")
    ax.set_title(f"MOS Score ({subject_id[:8]})")
    ax.set_ylim(y_min, y_max) 
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # save fig
    short_id = subject_id[:8]
    #save_path = SAVE_DIR / f"mos_{subject_id}.png"
    save_path = SAVE_DIR / f"mos_{short_id}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

