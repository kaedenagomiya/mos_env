import toybox
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# モデル名リスト
model_names_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
#rename_list = ['ljspeech', '', 'gradtfktts', 'gradtfk5tts', 'nix deter']
rename_list = ['GT', 'Standard-\nConv', 'TFKM3x3\n(proposed)', 'TFKM5x5\n(proposed)', 'Nix deter']

# カラーマップ
color_list = ['#ff6700', '#FF0000', '#85ACFF', '#245DFE', '#468585']
hatch_list = ['','','','','']
# グラフのY軸範囲
y_min = 1
y_max = 5.4

# ディレクトリ設定
DATA_DIR = Path('./results_evaluation')
SAVE_DIR = Path('./results_fig_mos')
SAVE_DIR.mkdir(exist_ok=True)  # 保存先フォルダを作成


# 各モデルの全被験者に対する平均と分散を格納するためのリスト
mos_means_all_subjects = {model_name: [] for model_name in model_names_list}
mos_std_all_subjects = {model_name: [] for model_name in model_names_list}
subject_counts = {model_name: 0 for model_name in model_names_list}  # 各モデルごとの被験者数を格納

# 実験に採用された被験者数（モデルごとに評価がある被験者の数）
subject_count_per_model = {model_name: set() for model_name in model_names_list}  # 各モデルの被験者IDのセット


# データフォルダ内の subject_id を取得
for subject_dir in DATA_DIR.iterdir():
    if not subject_dir.is_dir():
        continue  # ディレクトリでない場合スキップ

    subject_id = subject_dir.name
    eval_csv = subject_dir / 'audio_evaluations.csv'
    
    if not eval_csv.exists():
        print(f"Skipping {subject_id}: No evaluation file found.")
        continue  # ファイルがない場合スキップ

    # CSV 読み込み
    df_eval = pd.read_csv(eval_csv)

    # 行数チェック（99行未満ならスキップ）
    if len(df_eval) < 99:
        print(f"Skipping {subject_id}: Not enough data ({len(df_eval)} rows).")
        continue  

    # MOS 計算
    mos_means = df_eval.groupby("model_name")["evaluation"].mean()
    mos_means = mos_means.reindex(model_names_list)
    mos_std = df_eval.groupby("model_name")["evaluation"].std()
    mos_std = mos_std.reindex(model_names_list)

    print(mos_means)
    print(subject_dir)

    # 各モデルの平均値と分散を全被験者分に対してリストに追加
    for model_name in model_names_list:
        mos_means_all_subjects[model_name].append(mos_means[model_name])
        #mos_var_all_subjects[model_name].append(mos_std[model_name] ** 2)
        mos_std_all_subjects[model_name].append(mos_std[model_name])
        subject_counts[model_name] += 1

        # 実験に採用された被験者をカウント
        if model_name in df_eval['model_name'].values:
            subject_count_per_model[model_name].add(subject_id)


    # グラフ描画
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(mos_means.index, mos_means.values, yerr=mos_std.values, capsize=5, color=color_list, alpha=0.7)

    # MOS 値を棒の内部に表示
    for bar, mean in zip(bars, mos_means.values):
        text_y = max(y_min + 0.2, mean - 0.2)  # MOS値が小さすぎるときは調整
        ax.text(bar.get_x() + bar.get_width() / 2, text_y, f"{mean:.2f}", 
                ha='center', va='center', fontsize=12, fontweight='bold', color="white")

    # 軸ラベルとタイトル
    ax.set_ylabel("MOS Score")
    ax.set_xlabel("Model Name")
    ax.set_title(f"MOS Score ({subject_id})")
    ax.set_ylim(y_min, y_max)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    """
    # 画像を保存
    save_path = SAVE_DIR / f"mos_{subject_id}.pdf"
    #plt.savefig(save_path)
    #plt.close()
    toybox.make_mosfig(
        means=mos_means,
        std=mos_std,
        color_list=color_list,
        hatch_list=hatch_list,
        rename_list=rename_list,
        y_min=1,
        y_max=5.4,
        path_savefig=save_path,
        ylabel="MOS Score",
        xlabel="Model Name",
        title="MOS Score"
    )
    print(f"Saved: {save_path}")
    

# 各モデルに対する全被験者の平均と分散を計算
model_means = []
model_std_devs = []

print("\n----- Model Statistics -----")

for model_name in model_names_list:
    all_subject_means = mos_means_all_subjects[model_name]
    all_subject_std = mos_std_all_subjects[model_name]
    
    # 各モデルの全被験者の平均と分散
    mean_of_means = sum(all_subject_means) / len(all_subject_means)
    std_of_means = sum(all_subject_std) / len(all_subject_std)
    #std_of_means = np.sqrt(std_of_means) # std = np.sqrt(variance)
    
    model_means.append(mean_of_means)
    model_std_devs.append(std_of_means)

    num_subjects = len(subject_count_per_model[model_name])
    #print(f"{model_name} - Mean of MOS across all subjects: {mean_of_means:.2f}")
    #print(f"{model_name} - std of MOS across all subjects: {std_of_means:.2f}")
    print(f"{model_name} mean+std: {mean_of_means:.5f}+{std_of_means:.5f}")
    print(f"{model_name} - Number of subjects in experiment: {num_subjects}")
    print()



# convert to pd.Series
mosmean_pdseries = pd.Series(model_means, index=model_names_list)
mosstd_pdseries = pd.Series(model_std_devs, index=model_names_list)

save_path = SAVE_DIR / "average_mos_across_subjects.pdf"
toybox.make_mosfig(
    means=mosmean_pdseries,
    std=mosstd_pdseries,
    color_list=color_list,
    hatch_list=hatch_list,
    rename_list=rename_list,
    y_min=1,
    y_max=5.4,
    path_savefig=save_path,
    ylabel="MOS Score",
    xlabel="Model Name",
    title="MOS Score"
)

"""
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names_list, model_means, yerr=model_std_devs, capsize=5, color=color_list, alpha=0.7)

# MOS 値を棒の内部に表示
for bar, mean in zip(bars, model_means):
    #text_y = max(y_min + 0.2, mean - 0.2)  # MOS値が小さすぎるときは調整
    text_y = 1 + 0.2  # バーの内部に表示（中央やや上）
    text_color = "white" #if mean > (y_min + y_max) / 2 else "black"  # 明るさに応じて色を変える
    ax.text(bar.get_x() + bar.get_width() / 2, text_y, f"{mean:.2f}", 
            ha='center', va='center', fontsize=12, fontweight='bold', color="white")

# 軸ラベルとタイトル
ax.set_ylabel("MOS Score")
ax.set_xlabel("Model Name")
ax.set_title("Average MOS Score Across All Subjects")
ax.set_ylim(y_min, y_max)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# グラフを保存
save_path = SAVE_DIR / "average_mos_across_subjects.png"
plt.savefig(save_path)
plt.close()

print(f"Saved: {save_path}")
"""