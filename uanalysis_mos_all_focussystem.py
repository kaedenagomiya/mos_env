import toybox
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg #for welch anova
import scikit_posthocs as sp #for Steel-Dwass

flag_fig = False

model_names_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix_deter']
#rename_list = ['ljspeech', 'gradtts', 'gradtfktts', 'gradtfk5tts', 'nix deter']
rename_list = ['GT', 'Standard-\nConv', 'TFKM3x3\n(proposed)', 'TFKM5x5\n(proposed)', 'Nix deter']
df_columns_list = ['model_name', 'evaluation']

# カラーマップ
color_list = ['#ff6700', '#FF0000', '#85ACFF', '#245DFE', '#468585']
hatch_list = ['','','','','']
# グラフのY軸範囲
y_min = 1.9
y_max = 5.4

DATA_DIR = Path('./results_evaluation')
SAVE_DIR = Path('./results_fig_mos_focus_system')
save_path = SAVE_DIR / f"amos_focussystem.pdf"
SAVE_DIR.mkdir(exist_ok=True)  # 保存先フォルダを作成

#all_mos_score_focus_system = {model_name: [] for model_name in model_names_list}
#all_mos_score_focus_system = pd.DataFrame(columns=model_names_list)
df_all = pd.DataFrame(columns=df_columns_list)
#df_all = pd.DataFrame({'model_name': pd.Series(dtype='str'), 'evaluation': pd.Series(dtype='int')})
#print(all_mos_score_focus_system.to_dict())

# collect data
# データフォルダ内の subject_id を取得
for subject_dir in DATA_DIR.iterdir():
    #for subject_dir in [Path("results_evaluation/9567e112-c5c9-4e25-8e25-bc1bded7f2ff")]:
    #print(subject_dir)
    if not subject_dir.is_dir():
        continue  # ディレクトリでない場合スキップ

    subject_id = subject_dir.name
    eval_csv = subject_dir / 'audio_evaluations.csv'
    
    if not eval_csv.exists():
        print(f"Skipping {subject_id}: No evaluation file found.")
        continue  # ファイルがない場合スキップ

    # CSV 読み込み
    df_eval = pd.read_csv(eval_csv)
    #print(f"Read {len(df_eval)} rows from {eval_csv}")
    #print(df_eval)

    # 行数チェック（99行未満ならスキップ）
    if len(df_eval) < 99:
        print(f"Skipping {subject_id}: Not enough data ({len(df_eval)} rows).")
        continue
    
    # concat
    #print(f"Before concat, df_all shape: {df_all.shape}")
    df_all = pd.concat([df_all, df_eval], ignore_index=True)
    #print(f"After concat, df_all shape: {df_all.shape}")


# debug
"""
print(len(df_all))
print(df_all.columns)
print(f"Nan4all:{df_all.isnull().sum()}") 
df_all['model_name'] = df_all['model_name'].str.strip()
print(df_all['model_name'].unique())
print(df_all[df_all['model_name'].isnull()])
nan_model_data = df_all[df_all['model_name'].isnull()]
print("exp_version and user_id for rows with NaN model_name:")
print(nan_model_data[['exp_version', 'user_id', 'wav_path']])

print()
model_name_counts = df_all.groupby('model_name').size()
print(model_name_counts)
"""


#df_stats = df_all[["model_name", "evaluation"]]
#print(df_stats.columns)
#print(f"df_stats shape: {df_stats.shape}")
#print(f"each_model datalen: {df_stats.groupby('model_name')['evaluation'].size()}")
#df_all = df_all.set_index("model_name")
#df_ljspeech = df_stats.groupby("model_name").get_group("ljspeech")
#print(df_all.groupby("model_name").get_group("ljspeech"))


# analysis describe statics
model_stats = df_all.groupby('model_name')['evaluation'].agg(
    ['count', 'mean', 'std', 'min', 'max', 'median']
    )
#print(df_all.groupby('model_name'))
print(f"describe statics:\n {model_stats}")
#print(model_stats['mean'].iloc[0])
print(f"shape: {model_stats.shape}")
print()

# statistical test
#df_stats = df_all[["model_name", "evaluation"]].set_index('model_name')
#df_st = df_all[["model_name", "evaluation"]]
#print(df_st)

data_st = df_all.groupby("model_name")["evaluation"].apply(list).to_dict()

# すべてのキーを用意し、存在しないモデルには空リストをセット
all_models = ["ljspeech", "gradtts", "gradtfktts", "gradtfk5tts", "nix_deter"]
data_dict = {model: data_st.get(model, []) for model in all_models}

#print(data_dict)

#df_st = pd.DataFrame(
#    {name: group["evaluation"].astype(float).values for name, group in df_stats.groupby("model_name")}
#)

#print(df_st.groupby('model_name')['evaluation'].agg(
#    ['count', 'mean', 'std', 'min', 'max', 'median']
#    ))

df_sttest = pd.DataFrame(data_dict)
#print(df_sttest)
print("\n df statics")
print(df_sttest.shape)
print(df_sttest.describe())

mosmean_pdseries = df_sttest.mean() 
mosstd_pdseries = df_sttest.std()
#print(mosmean_pdseries)
#print(mosstd_pdseries)

save_path = SAVE_DIR / "amos_test.pdf"
if flag_fig == True:
    toybox.make_mosfig(
        means=mosmean_pdseries,
        std=mosstd_pdseries,
        color_list=color_list,
        hatch_list=hatch_list,
        rename_list=rename_list,
        y_min=y_min,
        y_max=y_max,
        path_savefig=save_path,
        ylabel="MOS Score",
        xlabel="Model Name",
        title="MOS Score"
    )

# -----------------------------------------------------------
# 1. 正規性の検定
print("### 正規性の検定 ###")
normality_results = {}
for model, values in data_dict.items():
    n = len(values)
    
    if n < 50:
        stat, p_value = stats.shapiro(values)  # Shapiro-Wilk
        test_name = "Shapiro-Wilk"
    elif 50 <= n < 300:
        stat, p_value = stats.anderson(values).statistic, min(stats.anderson(values).critical_values)
        test_name = "Anderson-Darling"
    else:
        stat, p_value = stats.kstest(values, 'norm')  # Kolmogorov-Smirnov
        test_name = "Kolmogorov-Smirnov"

    normality_results[model] = p_value
    print(f"{model}: {test_name} stat={stat}, p={p_value}")

# -----------------------------------------------------------
# 3. 群間の差の検定
print("\n### 群間の差の検定 ###")
all_normal = all(p > 0.05 for p in normality_results.values())
print(all_normal)

all_normal = p_value > 0.05
if all_normal==True:
    print("have normality.")
else:
    print("Not have normality.")

# -----------------------------------------------------------
# 2. 分散の等質性の検定
print("\n### 分散の等質性の検定 (Levene test) ###")
stat, p_value = stats.levene(*data_dict.values())
equal_variance = p_value > 0.05
print(f"Levene test: W={stat:.4f}, p={p_value} (Equal variance: {equal_variance})")
if equal_variance==True:
    print("have equal variance.")
else:
    print("Not have equal variance.")


# -----------------------------------------------------------
# ここで、正規性がなくてもKruskall-Wallis検定を行います
print("\n### Kruskal-Wallis検定 ###")
stat, p_value = stats.kruskal(*data_dict.values())  # Kruskal-Wallis
print(f"Kruskal-Wallis test: H={stat:.4f}, p={p_value}")
stat_diff = p_value < 0.05  # Kruskal-Wallisのp値が有意かどうかを確認
if stat_diff==True:
    print("have statistical difference")
else:
    print("Not have statistical difference")


if all_normal==True and equal_variance==True:
    print("\n### ANOVA検定 ###")
    stat, p_value = stats.f_oneway(*data_dict.values())  # ANOVA
    print(f"ANOVA: F={stat:.4f}, p={p_value}")
    use_anova = p_value < 0.05
elif all_normal==True and equal_variance==False:
    print("\n### Welch ANOVA検定 ###")
    df = pd.DataFrame(
        [(model, score) for model, values in data_dict.items() for score in values],
        columns=["model", "score"]
    )
    welch_anova_result = pg.welch_anova(dv="score", between="model", data=df)
    stat, p_value = welch_anova_result["F"].values[0], welch_anova_result["p-unc"].values[0]
    print(f"Welch ANOVA: F={stat:.4f}, p={p_value}")
    use_anova = p_value < 0.05
else:
    print("Not have normality and equal_variance")


# -----------------------------------------------------------
# 4. Tukey HSD or Steel-Dwass
# all_normal = p_value > 0.05
# stat_diff = p_value < 0.05
# equal_variance = p_value > 0.05
if all_normal==True and equal_variance==True:
    print("\n### Tukey HSD 検定 (Post-hoc test) ###")
    model_labels = []
    scores = []
    
    for model, values in data_dict.items():
        model_labels.extend([model] * len(values))
        scores.extend(values)
    
    tukey_result = pairwise_tukeyhsd(endog=scores, groups=model_labels, alpha=0.05)
    print(tukey_result)
if all_normal==True and equal_variance==False:
    #if not all_normal or not equal_variance:  # 正規性がないか等分散性がない場合
    print("\n### ゲームズ・ハウエル検定 ###")
    # データを適切な形式に変換
    df = pd.DataFrame(
        [(model, score) for model, values in data_dict.items() for score in values],
        columns=["model", "score"]
    )
    gameshowell_result = pg.pairwise_gameshowell(dv="score", between="model", data=df)
    print(gameshowell_result)
    
    # ゲームズ・ハウエル検定結果を確認して有意差があるかを表示
    print("\n### ゲームズ・ハウエル検定結果（有意差の有無） ###")
    significant_diff_gameshowell = gameshowell_result['pval'] < 0.05
    print(significant_diff_gameshowell)

elif all_normal==False:
    # Kruskal-Wallisで有意差があった場合、Steel-Dwass検定を実施
    print("\n### Steel-Dwass 検定（Kruskal-Wallis後の多重比較） ###")
    data = [values for values in data_dict.values()]
    posthoc_result = sp.posthoc_dscf(data)
    print(posthoc_result)
    # Steel-Dwass 検定の結果を確認して有意差があるかを表示
    significant_diff = (posthoc_result < 0.05)
    print("\n### Steel-Dwass 検定結果（有意差の有無） ###")
    print(significant_diff)
else:
    print("\(--)/don't know")



"""
# 1. 正規性の検定
print("### 正規性の検定 ###")
normality_results = {}
for model, values in data_dict.items():
    n = len(values)
    
    if n < 50:
        stat, p_value = stats.shapiro(values)  # Shapiro-Wilk
        test_name = "Shapiro-Wilk"
    elif 50 <= n < 300:
        stat, p_value = stats.anderson(values).statistic, min(stats.anderson(values).critical_values)
        test_name = "Anderson-Darling"
    else:
        stat, p_value = stats.kstest(values, 'norm')  # Kolmogorov-Smirnov
        test_name = "Kolmogorov-Smirnov"

    normality_results[model] = p_value
    print(f"{model}: {test_name} stat={stat:.4f}, p={p_value:.4f}")

# 2. 分散の等質性の検定
print("\n### 分散の等質性の検定 (Levene test) ###")
stat, p_value = stats.levene(*data_dict.values())
equal_variance = p_value > 0.05
print(f"Levene test: W={stat:.4f}, p={p_value} (Equal variance: {equal_variance})")

# 3. 群間の差の検定
print("\n### 群間の差の検定 ###")
all_normal = all(p > 0.05 for p in normality_results.values())

if all_normal:
    if equal_variance:
        stat, p_value = stats.f_oneway(*data_dict.values())  # ANOVA
        print(f"ANOVA: F={stat:.4f}, p={p_value:.4f}")
        use_anova = p_value < 0.05
    else:
        df = pd.DataFrame(
            [(model, score) for model, values in data_dict.items() for score in values],
            columns=["model", "score"]
        )
        welch_anova_result = pg.welch_anova(dv="score", between="model", data=df)
        stat, p_value = welch_anova_result["F"].values[0], welch_anova_result["p-unc"].values[0]
        print(f"Welch ANOVA: F={stat:.4f}, p={p_value:.4f}")
        use_anova = p_value < 0.05
else:
    stat, p_value = stats.kruskal(*data_dict.values())  # Kruskal-Wallis
    print(f"Kruskal-Wallis test: H={stat:.4f}, p={p_value:.4f}")
    use_anova = False

# 4. Tukey HSD or Steel-Dwass
if use_anova:
    print("\n### Tukey HSD 検定 (Post-hoc test) ###")
    model_labels = []
    scores = []
    
    for model, values in data_dict.items():
        model_labels.extend([model] * len(values))
        scores.extend(values)
    
    tukey_result = pairwise_tukeyhsd(endog=scores, groups=model_labels, alpha=0.05)
    print(tukey_result)
else:
    # クラスカル・ウォリスで有意差が出た場合、Steel-Dwass 検定を実施
    print("\n### Steel-Dwass 検定（Kruskal-Wallis後の多重比較） ###")
    data = [values for values in data_dict.values()]
    posthoc_result = sp.posthoc_dscf(data)
    print(posthoc_result)

    print("\nANOVA で有意差がなかったか、正規分布でないため Tukey HSD を実施しません。")
"""
"""
# Normality Test(Shapiro-Wilk test)
print("Shapiro-Wilk test for normality:")
for col in df_st.columns:
    print(df_st[col])
    stat, p_value = stats.shapiro(df_st[col])
    print(f"Column {col} - Shapiro-Wilk Test: Stat={stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print(f"Column {col} is normally distributed.\n")
    else:
        print(f"Column {col} is not normally distributed.\n")


# Test of Equal Variance (Levene Test)
levene_stat, levene_pval = stats.levene(*df_st.values())
print(f"\nLevene test for equal variances: p-value = {levene_pval:.5f}")

# ANOVA検定（モデル間の平均値に差があるか）
anova_stat, anova_pval = stats.f_oneway(*df_st.values())
print(f"\nANOVA test: p-value = {anova_pval:.5f}")

# multiple comparisons（Tukey HSD）
all_data = []
all_labels = []
for model, values in df_st.items():
    all_data.extend(values)
    all_labels.extend([model] * len(values))

tukey_result = pairwise_tukeyhsd(endog=all_data, groups=all_labels, alpha=0.05)
print("\nTukey HSD test results:")
print(tukey_result)
"""



"""

# 特定のモデルに絞って評価スコアを表示
#ljspeech_data = df_stats[df_stats['model_name'] == 'ljspeech']['evaluation']
#print(ljspeech_data)
"""
"""
# モデルごとの平均評価を計算
model_stats = df_all.groupby('model_name')['evaluation'].agg(['mean', 'std', 'min', 'max', 'median'])
print(model_stats)
import seaborn as sns
import matplotlib.pyplot as plt

# 箱ひげ図を作成
plt.figure(figsize=(10, 6))
sns.boxplot(x='model_name', y='evaluation', data=df_all, palette='Set2')
plt.xticks(rotation=45)  # x軸のラベルを回転させる
plt.title('Evaluation Scores by Model')
plt.show()

# モデルごとの評価のカウント
model_counts = df_all.groupby('model_name')['evaluation'].count()
print(model_counts)

# 各モデル名ごとに評価の割合を計算
model_eval_percentages = df_all.groupby('model_name')['evaluation'].value_counts(normalize=True).unstack() * 100
print(model_eval_percentages)

# 特定のモデルに絞って評価スコアを表示
ljspeech_data = df_all[df_all['model_name'] == 'ljspeech']
print(ljspeech_data)
"""

"""
print(len(df_all))
print(df_all.groupby("model_name").get_group("ljspeech"))
print(df_all.groupby("model_name").get_group("ljspeech")["evaluation"].size)
print(df_all.groupby("model_name").get_group("ljspeech").shape)
print(df_all.groupby("model_name").get_group("gradtts").shape)
print(df_all.groupby("model_name").get_group("gradtfktts").shape)
print(df_all.groupby("model_name").get_group("gradtfk5tts").shape)
print(df_all.groupby("model_name").get_group("nix_deter").shape)
"""
