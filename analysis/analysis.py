import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode, ttest_ind, pearsonr, kruskal, mannwhitneyu
from data import id_data, cs_data, id_qualitative, cs_qualitative

# Convert data to DataFrames
id_df = pd.DataFrame(id_data)
cs_df = pd.DataFrame(cs_data)

# Combine ID and CS students into one DataFrame for general analysis
all_students_df = pd.concat([id_df, cs_df], ignore_index=True)

# ------------------- Helper Function -------------------
# Descriptive statistics function
def descriptive_stats(df, column):
    mean = df[column].mean()
    median = df[column].median()
    mode_result = mode(df[column], nan_policy="omit")
    std = df[column].std()
    return {"mean": mean, "median": median, "mode": mode_result, "std": std}

# Display Descriptive Statistics
print("\n--- Descriptive Statistics ---")
for group, df in [("ID Students", id_df), ("CS Students", cs_df)]:
    print(f"\n{group}:")
    for col in ["initial_scores", "ml_pipelines", "bayes_rule", "perceptrons"]:
        stats = descriptive_stats(df, col)
        print(f"  {col.capitalize()}: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Mode={stats['mode']}, Std={stats['std']:.2f}")

# ---------------------- Sub-question 1: Comparison of Initial Math Scores ----------------------
# # T-test for comparing initial math scores
# t_stat, p_value = ttest_ind(id_df["initial_scores"], cs_df["initial_scores"])
# print("\nSub-question 1: Initial Math Scores Comparison")
# print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")

# # Boxplot for math scores
# plt.figure(figsize=(10, 6))
# data = [id_df["initial_scores"], cs_df["initial_scores"]]
# plt.boxplot(data, labels=["ID Students", "CS Students"], patch_artist=True)
# plt.title("Boxplot: Prior Math Knowledge Scores")
# plt.ylabel("Weighted Scores (%)")
# plt.show()

# Perform Mann-Whitney U test (alternative to T-test for non-normal distributions)
stat, p_value = mannwhitneyu(id_df["initial_scores"], cs_df["initial_scores"])
print("\nSub-question 1: Initial Math Scores Comparison (Mann-Whitney U Test)")
print(f"U-statistic: {stat:.2f}, P-value: {p_value:.4f}")

# Boxplot for math scores
plt.figure(figsize=(10, 6))
data = [id_df["initial_scores"], cs_df["initial_scores"]]
plt.boxplot(data, labels=["ID Students", "CS Students"], patch_artist=True)
plt.title("Boxplot: Prior Math Knowledge Scores")
plt.ylabel("Weighted Scores (%)")
plt.show()

id_maths_background = len(id_qualitative["maths_background"])
cs_maths_background = len(cs_qualitative["maths_background"])

print("Number of courses took previously, taking now, and activities related to mathematics")
print(f"\tID: {id_maths_background}")
print(f"\tCS: {cs_maths_background}")

# ---------------------- Sub-question 2: Correlation Analysis ----------------------
# Correlation between initial scores and final ML scores for all students
print("\nSub-question 2: Correlation Analysis")
corr_ml_pipeline, pval_ml_pipeline = pearsonr(all_students_df["initial_scores"], all_students_df["ml_pipelines"])
corr_bayes_rule, pval_bayes_rule = pearsonr(all_students_df["initial_scores"], all_students_df["bayes_rule"])
corr_perceptrons, pval_perceptrons = pearsonr(all_students_df["initial_scores"], all_students_df["perceptrons"])
print(f"All Students:")
print(f"  - ML Pipelines: r = {corr_ml_pipeline:.2f}, p = {pval_ml_pipeline:.4f}")
print(f"  - Bayes Rule: r = {corr_bayes_rule:.2f}, p = {pval_bayes_rule:.4f}")
print(f"  - Perceptrons: r = {corr_perceptrons:.2f}, p = {pval_perceptrons:.4f}")

# Visualization: Scatter plots for correlation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
topics = ["ml_pipelines", "bayes_rule", "perceptrons"]
titles = ["ML Pipelines", "Bayes Rule", "Perceptrons"]

for ax, topic, title in zip(axes, topics, titles):
    sns.regplot(
        x="initial_scores", y=topic, data=all_students_df, ax=ax, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}
    )
    ax.set_title(f"Initial Math Scores vs {title}")
    ax.set_xlabel("Initial Math Scores")
    ax.set_ylabel(f"{title} Scores")

plt.tight_layout()
plt.show()

# ---------------------- Sub-question 3: Performance Across ML Topics ----------------------
# Combine faculty labels for group comparison
id_topics = id_df.melt(id_vars=["initial_scores"], value_vars=["ml_pipelines", "bayes_rule", "perceptrons"], var_name="topic", value_name="score")
id_topics["faculty"] = "ID"

cs_topics = cs_df.melt(id_vars=["initial_scores"], value_vars=["ml_pipelines", "bayes_rule", "perceptrons"], var_name="topic", value_name="score")
cs_topics["faculty"] = "CS"

all_topics = pd.concat([id_topics, cs_topics], ignore_index=True)

# Kruskal-Wallis test by faculty for each topic
print("\nSub-question 3: Faculty Performance on ML Topics")
for topic in ["ml_pipelines", "bayes_rule", "perceptrons"]:
    id_scores = id_df[topic]
    cs_scores = cs_df[topic]
    h_stat, p_val = kruskal(id_scores, cs_scores)
    print(f"Topic: {topic.capitalize()} - H-statistic = {h_stat:.2f}, P-value = {p_val:.4f}")

# Visualization: Bar plot for faculty performance
plt.figure(figsize=(10, 6))
sns.barplot(data=all_topics, x="topic", y="score", hue="faculty", ci="sd", palette="muted")
plt.title("Performance Across ML Topics by Faculty")
plt.xlabel("ML Topics")
plt.ylabel("Scores")
plt.legend(title="Faculty")
plt.show()

# ---------------------- Sub-question 4: Qualitative Patterns ----------------------
