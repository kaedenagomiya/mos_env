import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

data = {
    "subject_id": [
        "9567e112", "81a7080b", "b33cc0b2", "fa2d5dea", "1276e5c9", "d14b1fd9",
        "0136e521", "531d5a82", "3b24700b", "5f558e4e", "df1d30e3", "9016a613"
    ],
    "ljspeech": [4.47, 3.85, 4.05, 4.8, 3.2, 5.0, 4.9, 4.65, 4.25, 3.95, 4.85, 4.95],
    "gradtts": [4.65, 3.85, 4.0, 4.05, 3.8, 4.35, 4.35, 4.2, 4.05, 3.4, 4.5, 4.4],
    "gradtfktts": [3.95, 3.85, 3.15, 3.65, 3.4, 3.85, 3.5, 4.0, 4.1, 3.1, 4.3, 4.4],
    "gradtfk5tts": [4.2, 3.7, 3.6, 3.6, 3.45, 4.35, 3.85, 4.35, 4.35, 3.45, 4.4, 4.3],
    "nix_deter": [3.95, 3.95, 2.3, 3.4, 2.6, 3.85, 3.35, 3.6, 4.25, 3.3, 3.7, 4.2],
}
df = pd.DataFrame(data)

# [seq] test
# Normality Test(Shapiro-Wilk test)
print("Shapiro-Wilk test results:")
for model in ["ljspeech", "gradtts", "gradtfktts", "gradtfk5tts", "nix_deter"]:
    stat, p_value = stats.shapiro(df[model])
    print(f"{model}: W={stat:.4f}, p={p_value:.4f}")

# Test of Equal Variance (Levene Test)
levene_stat, levene_p = stats.levene(
    df["ljspeech"], df["gradtts"], df["gradtfktts"], df["gradtfk5tts"], df["nix_deter"]
)
print("\nLevene test results:")
print(f"Levene statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}")

# convert df
df_long = df.melt(id_vars=["subject_id"], var_name="model", value_name="score")

# ANOVA
anova_result = stats.f_oneway(
    df["ljspeech"], df["gradtts"], df["gradtfktts"], df["gradtfk5tts"], df["nix_deter"]
)

print("\nANOVA results:")
print(f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4f}")

# 多重比較（Tukey HSD）
tukey_result = pairwise_tukeyhsd(df_long["score"], df_long["model"], alpha=0.05)

print("\nTukey HSD results:")
print(tukey_result)

# -------------------------------------------------------------------------
print("If there is no normality")
# Kruskal-Wallis検定（ノンパラメトリックANOVA）
kruskal_stat, kruskal_p = stats.kruskal(
    df["ljspeech"], df["gradtts"], df["gradtfktts"], df["gradtfk5tts"], df["nix_deter"]
)

print("\nKruskal-Wallis test results（non parametric ANOVA）:")
print(f"K-statistic: {kruskal_stat:.4f}, p-value: {kruskal_p:.4f}")

# Dunnの多重比較検定（p値をBonferroni補正）
dunn_results = sp.posthoc_dunn(df_long, val_col="score", group_col="model", p_adjust="bonferroni")

print("\nDunn's multiple comparison test results (Bonferroni correction applied):")
print(dunn_results)


# ============================================================================
"""
print("Delta of Cliff")

# Cliff's Delta
#|delta| = 0.00 -> ほぼ差なし
#|delta| = 0.15 -> 小さい差
#|delta| = 0.34 -> 中程度の差
#|delta| = 0.48 -> 大きな差

# Dunn's Multiple Comparison Test (Bonferroni correction)
df_long_cliff = df.melt(id_vars=["subject_id"], var_name="group", value_name="score")
dunn_results = sp.posthoc_dunn(df_long_cliff, val_col="score", group_col="group", p_adjust='bonferroni')

# 有意差のあるペアを抽出
alpha = 0.05
significant_pairs = [(g1, g2) for g1, g2 in itertools.combinations(df_long_cliff["group"].unique(), 2) if dunn_results.loc[g1, g2] < alpha]

# Cliffs Delta を計算する関数
def cliffs_delta(x, y):
    #Cliffs Delta を計算する関数
    n_x, n_y = len(x), len(y)
    count = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                count += 1
            elif xi < yj:
                count -= 1
    delta = count / (n_x * n_y)
    return delta

# 有意差のあるペアに対して Cliff's Delta を計算
effect_sizes = {}
for g1, g2 in significant_pairs:
    # 各グループにおけるスコアを取得
    scores_g1 = df[df["subject_id"].isin(df_long_cliff[df_long_cliff["group"] == g1]["subject_id"])][g1].values
    scores_g2 = df[df["subject_id"].isin(df_long_cliff[df_long_cliff["group"] == g2]["subject_id"])][g2].values
    delta = cliffs_delta(scores_g1, scores_g2)
    effect_sizes[(g1, g2)] = delta

# 結果表示
print("Dunn's Multiple Comparison Test (Bonferroni correction):")
print(dunn_results)

print("\nSignificant Pairs and Their Cliff's Delta:")
for (g1, g2), delta in effect_sizes.items():
    print(f"{g1} vs. {g2}: Cliff's Delta = {delta:.3f}")
"""