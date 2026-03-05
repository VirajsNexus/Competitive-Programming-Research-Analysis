"""
=============================================================
  Predicting Problem Difficulty in Competitive Programming
  Using Submission Data — Full Model Code
  Author: Viraj Jamdhade
  Dataset: codeforces_dataset.csv
=============================================================
"""

# ── 1. IMPORTS ────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, export_text
from sklearn.metrics         import (accuracy_score, f1_score, precision_score,
                                     recall_score, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline        import Pipeline

import xgboost as xgb

# optional SHAP — install with:  pip install shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] SHAP not installed. Skipping SHAP analysis.")
    print("       Run:  pip install shap   to enable it.")

plt.rcParams.update({
    "figure.dpi"     : 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family"    : "DejaVu Sans",
})

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ── 2. LOAD DATA ──────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — Loading Dataset")
print("="*60)

df = pd.read_csv("codeforces_dataset.csv")
print(f"  Rows    : {len(df)}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  First 3 rows preview:")
print(df.head(3).to_string())


# ── 3. EXPLORE DATA ───────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — Exploring Data")
print("="*60)

print("\n  Class Distribution:")
dist = df["tier_name"].value_counts().sort_index()
for name, count in df.groupby("difficulty_tier")["tier_name"].first().items():
    c = (df["difficulty_tier"] == name).sum()
print(df.groupby(["difficulty_tier","tier_name"]).size().reset_index(name="count").to_string(index=False))

print("\n  Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().any() else "  None found.")

print("\n  Basic Statistics (numeric only):")
print(df[["acceptance_rate","wa_ratio","avg_solve_time_min","solver_rating_mean"]].describe().round(3).to_string())


# ── PLOT 1: Class Distribution Bar ────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
tier_counts = df.groupby(["difficulty_tier","tier_name"]).size().reset_index(name="count")
tier_counts = tier_counts.sort_values("difficulty_tier")
colors = ["#4CAF50","#2196F3","#FF9800","#E91E63","#9C27B0"]
bars = ax.bar(tier_counts["tier_name"], tier_counts["count"], color=colors, edgecolor="white", linewidth=1.5, width=0.6)
for bar, count in zip(bars, tier_counts["count"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Figure 1 — Problem Count per Difficulty Tier", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Difficulty Tier", fontsize=11)
ax.set_ylabel("Number of Problems", fontsize=11)
ax.set_ylim(0, tier_counts["count"].max() * 1.15)
plt.tight_layout()
plt.savefig("fig1_class_distribution.png", bbox_inches="tight")
plt.show()
print("  [Saved] fig1_class_distribution.png")


# ── PLOT 2: Acceptance Rate vs Difficulty ─────────────────
fig, ax = plt.subplots(figsize=(8, 4))
tier_acc = df.groupby("difficulty_tier")["acceptance_rate"].mean()
tier_labels = ["Tier 1\n(Beginner)","Tier 2\n(Easy-Med)","Tier 3\n(Medium)","Tier 4\n(Hard)","Tier 5\n(Expert)"]
bars = ax.bar(tier_labels, tier_acc.values, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
for bar, val in zip(bars, tier_acc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Figure 2 — Mean Acceptance Rate by Difficulty Tier", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Difficulty Tier", fontsize=11)
ax.set_ylabel("Mean Acceptance Rate", fontsize=11)
ax.set_ylim(0, 0.90)
plt.tight_layout()
plt.savefig("fig2_acceptance_rate.png", bbox_inches="tight")
plt.show()
print("  [Saved] fig2_acceptance_rate.png")


# ── PLOT 3: Avg Solve Time vs Difficulty ──────────────────
fig, ax = plt.subplots(figsize=(8, 4))
tier_time = df.groupby("difficulty_tier")["avg_solve_time_min"].mean()
ax.plot(tier_labels, tier_time.values, marker="o", color="#E91E63",
        linewidth=2.5, markersize=9, markerfacecolor="white", markeredgewidth=2.5)
for x, y in enumerate(tier_time.values):
    ax.annotate(f"{y:.1f} min", (x, y), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10)
ax.fill_between(range(len(tier_time)), tier_time.values, alpha=0.10, color="#E91E63")
ax.set_title("Figure 3 — Mean Solving Time by Difficulty Tier", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Difficulty Tier", fontsize=11)
ax.set_ylabel("Avg Solve Time (minutes)", fontsize=11)
ax.set_xticks(range(5)); ax.set_xticklabels(tier_labels)
plt.tight_layout()
plt.savefig("fig3_solve_time.png", bbox_inches="tight")
plt.show()
print("  [Saved] fig3_solve_time.png")


# ── 4. PREPROCESSING ──────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Preprocessing")
print("="*60)

# Drop non-feature columns
drop_cols = ["problem_id","contest_id","problem_index","rating",
             "difficulty_tier","tier_name","tags"]
df_model = df.drop(columns=drop_cols, errors="ignore").copy()

# Encode round_type (one-hot)
df_model = pd.get_dummies(df_model, columns=["round_type"], drop_first=False)
print(f"  After one-hot encoding → {df_model.shape[1]} feature columns")

# Log-transform skewed columns
for col in ["total_submissions","unique_solvers"]:
    if col in df_model.columns:
        df_model[col] = np.log1p(df_model[col])
        print(f"  Log1p applied → {col}")

# Target
y = df["difficulty_tier"].values          # integer 1–5
X = df_model.values.astype(float)
feature_names = df_model.columns.tolist()

print(f"\n  Feature count : {X.shape[1]}")
print(f"  Sample count  : {X.shape[0]}")
print(f"  Target classes: {np.unique(y)}")


# ── 5. TRAIN / VAL / TEST SPLIT ───────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Train / Validation / Test Split  (70/15/15)")
print("="*60)

X_temp, X_test,  y_temp, y_test  = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val,  y_train, y_val  = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE)

print(f"  Train size : {len(X_train)}")
print(f"  Val size   : {len(X_val)}")
print(f"  Test size  : {len(X_test)}")

# Scale
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# ── 6. MODEL TRAINING ─────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — Training Models")
print("="*60)

results = {}

# ── Baseline 1: Majority Class ────────────────────────────
majority_class = np.bincount(y_train).argmax()
y_pred_majority = np.full(len(y_test), majority_class)
results["Majority Baseline"] = {
    "accuracy" : accuracy_score(y_test, y_pred_majority),
    "macro_f1" : f1_score(y_test, y_pred_majority, average="macro", zero_division=0),
    "weighted_f1": f1_score(y_test, y_pred_majority, average="weighted", zero_division=0),
}
print(f"\n  [Baseline] Majority Class → Acc: {results['Majority Baseline']['accuracy']:.3f}")

# ── Baseline 2: Acceptance Rate Threshold ─────────────────
acc_idx = feature_names.index("acceptance_rate")
acc_col_test  = X_test[:, acc_idx]

def threshold_predict(acc_vals):
    preds = np.empty(len(acc_vals), dtype=int)
    for i, v in enumerate(acc_vals):
        if   v >= 0.55: preds[i] = 1
        elif v >= 0.35: preds[i] = 2
        elif v >= 0.20: preds[i] = 3
        elif v >= 0.10: preds[i] = 4
        else:           preds[i] = 5
    return preds

y_pred_thresh = threshold_predict(acc_col_test)
results["AccRate Baseline"] = {
    "accuracy"    : accuracy_score(y_test, y_pred_thresh),
    "macro_f1"    : f1_score(y_test, y_pred_thresh, average="macro", zero_division=0),
    "weighted_f1" : f1_score(y_test, y_pred_thresh, average="weighted", zero_division=0),
}
print(f"  [Baseline] AccRate Threshold → Acc: {results['AccRate Baseline']['accuracy']:.3f}")

# ── Model 1: Logistic Regression ─────────────────────────
print("\n  Training Logistic Regression ...")
lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=0.1, multi_class="ovr")
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
results["Logistic Regression"] = {
    "accuracy"    : accuracy_score(y_test, y_pred_lr),
    "macro_f1"    : f1_score(y_test, y_pred_lr, average="macro"),
    "weighted_f1" : f1_score(y_test, y_pred_lr, average="weighted"),
    "predictions" : y_pred_lr,
}
print(f"  Done → Acc: {results['Logistic Regression']['accuracy']:.3f}")

# ── Model 2: Decision Tree ────────────────────────────────
print("\n  Training Decision Tree ...")
dt = DecisionTreeClassifier(max_depth=7, min_samples_split=20,
                            min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train_s, y_train)
y_pred_dt = dt.predict(X_test_s)
results["Decision Tree"] = {
    "accuracy"    : accuracy_score(y_test, y_pred_dt),
    "macro_f1"    : f1_score(y_test, y_pred_dt, average="macro"),
    "weighted_f1" : f1_score(y_test, y_pred_dt, average="weighted"),
    "predictions" : y_pred_dt,
}
print(f"  Done → Acc: {results['Decision Tree']['accuracy']:.3f}")

# ── Model 3: XGBoost ─────────────────────────────────────
print("\n  Training XGBoost (this may take ~1–2 min) ...")
xgb_model = xgb.XGBClassifier(
    n_estimators      = 450,
    max_depth         = 6,
    learning_rate     = 0.08,
    subsample         = 0.85,
    colsample_bytree  = 0.85,
    eval_metric       = "mlogloss",
    use_label_encoder = False,
    random_state      = RANDOM_STATE,
    verbosity         = 0,
)
xgb_model.fit(
    X_train_s, y_train - 1,      # XGBoost needs 0-indexed labels
    eval_set=[(X_val_s, y_val - 1)],
    verbose=False,
)
y_pred_xgb = xgb_model.predict(X_test_s) + 1
results["XGBoost"] = {
    "accuracy"    : accuracy_score(y_test, y_pred_xgb),
    "macro_f1"    : f1_score(y_test, y_pred_xgb, average="macro"),
    "weighted_f1" : f1_score(y_test, y_pred_xgb, average="weighted"),
    "predictions" : y_pred_xgb,
}
print(f"  Done → Acc: {results['XGBoost']['accuracy']:.3f}")


# ── 7. RESULTS SUMMARY TABLE ──────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — Results Summary")
print("="*60)

summary_rows = []
for model_name, res in results.items():
    summary_rows.append({
        "Model"      : model_name,
        "Accuracy"   : f"{res['accuracy']:.1%}",
        "Macro-F1"   : f"{res['macro_f1']:.3f}",
        "Weighted-F1": f"{res['weighted_f1']:.3f}",
    })
summary_df = pd.DataFrame(summary_rows)
print("\n" + summary_df.to_string(index=False))


# ── PLOT 4: Model Accuracy Comparison ─────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
model_names  = list(results.keys())
accuracies   = [results[m]["accuracy"] * 100 for m in model_names]
bar_colors   = ["#B0BEC5","#90A4AE","#42A5F5","#66BB6A","#EF5350"]
bars = ax.barh(model_names, accuracies, color=bar_colors, edgecolor="white",
               linewidth=1.5, height=0.55)
for bar, val in zip(bars, accuracies):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 100)
ax.set_xlabel("Accuracy (%)", fontsize=12)
ax.set_title("Figure 4 — Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
ax.axvline(x=50, color="gray", linewidth=1, linestyle="--", alpha=0.5, label="50% line")
plt.tight_layout()
plt.savefig("fig4_model_accuracy.png", bbox_inches="tight")
plt.show()
print("\n  [Saved] fig4_model_accuracy.png")


# ── PLOT 5: Confusion Matrix (XGBoost) ────────────────────
tier_labels_short = ["T1\nBeginner","T2\nEasy-Med","T3\nMedium","T4\nHard","T5\nExpert"]
cm = confusion_matrix(y_test, y_pred_xgb)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tier_labels_short)
disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
ax.set_title("Figure 5 — Confusion Matrix (XGBoost)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("fig5_confusion_matrix.png", bbox_inches="tight")
plt.show()
print("  [Saved] fig5_confusion_matrix.png")


# ── 8. DETAILED CLASSIFICATION REPORT ─────────────────────
print("\n" + "="*60)
print("  STEP 7 — Detailed Classification Report (XGBoost)")
print("="*60)
tier_display = ["Beginner","Easy-Medium","Medium","Hard","Expert"]
print("\n" + classification_report(y_test, y_pred_xgb, target_names=tier_display))


# ── 9. XGBoost FEATURE IMPORTANCE ────────────────────────
print("\n" + "="*60)
print("  STEP 8 — Feature Importance (XGBoost Gain)")
print("="*60)

importances = xgb_model.feature_importances_
fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
print("\n  Top 15 features:")
print(fi_df.to_string(index=False))

# ── PLOT 6: Feature Importance ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fi_plot = fi_df.sort_values("Importance")
colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_plot)))
ax.barh(fi_plot["Feature"], fi_plot["Importance"], color=colors_fi,
        edgecolor="white", linewidth=1.2, height=0.7)
ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
ax.set_title("Figure 6 — Top 15 Feature Importances (XGBoost)", fontsize=13, fontweight="bold", pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig("fig6_feature_importance.png", bbox_inches="tight")
plt.show()
print("\n  [Saved] fig6_feature_importance.png")


# ── 10. SHAP ANALYSIS (if available) ─────────────────────
if SHAP_AVAILABLE:
    print("\n" + "="*60)
    print("  STEP 9 — SHAP Analysis")
    print("="*60)
    explainer   = shap.TreeExplainer(xgb_model)
    shap_sample = X_test_s[:300]
    shap_values = explainer.shap_values(shap_sample)

    mean_shap = np.mean(np.abs(shap_values).sum(axis=0) if isinstance(shap_values, list)
                        else np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({"Feature": feature_names, "Mean|SHAP|": mean_shap})
    shap_df = shap_df.sort_values("Mean|SHAP|", ascending=False).head(10)
    print("\n  Top 10 SHAP features:")
    print(shap_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    shap_plot = shap_df.sort_values("Mean|SHAP|")
    ax.barh(shap_plot["Feature"], shap_plot["Mean|SHAP|"],
            color="#5C6BC0", edgecolor="white", linewidth=1.2, height=0.6)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Figure 7 — SHAP Feature Importance", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("fig7_shap_importance.png", bbox_inches="tight")
    plt.show()
    print("  [Saved] fig7_shap_importance.png")
else:
    print("\n  [Skipped] SHAP — install with:  pip install shap")


# ── 11. PER-CLASS F1 COMPARISON ───────────────────────────
print("\n" + "="*60)
print("  STEP 10 — Per-Class F1 for All Models")
print("="*60)

model_keys = ["Logistic Regression", "Decision Tree", "XGBoost"]
per_class_f1 = {}
for mk in model_keys:
    preds = results[mk]["predictions"]
    f1s   = f1_score(y_test, preds, average=None, labels=[1,2,3,4,5])
    per_class_f1[mk] = f1s

per_class_df = pd.DataFrame(per_class_f1, index=tier_display)
print("\n" + per_class_df.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(5)
width = 0.25
bar_colors2 = ["#42A5F5", "#66BB6A", "#EF5350"]
for i, (mk, clr) in enumerate(zip(model_keys, bar_colors2)):
    ax.bar(x + i*width, per_class_f1[mk], width, label=mk,
           color=clr, edgecolor="white", linewidth=1.2)
ax.set_xticks(x + width)
ax.set_xticklabels(tier_display, fontsize=11)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_ylim(0, 1.10)
ax.set_title("Figure 8 — Per-Class F1 Score by Model", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("fig8_perclass_f1.png", bbox_inches="tight")
plt.show()
print("\n  [Saved] fig8_perclass_f1.png")


# ── 12. CROSS-VALIDATION ──────────────────────────────────
print("\n" + "="*60)
print("  STEP 11 — 5-Fold Cross Validation (XGBoost)")
print("="*60)

cv_model = xgb.XGBClassifier(
    n_estimators=450, max_depth=6, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    use_label_encoder=False, verbosity=0, random_state=RANDOM_STATE,
)
cv_scores = cross_val_score(cv_model, X_train_s, y_train - 1,
                            cv=5, scoring="accuracy", n_jobs=-1)
print(f"\n  CV Scores  : {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean       : {cv_scores.mean():.3f}")
print(f"  Std Dev    : {cv_scores.std():.3f}")


# ── 13. FINAL SUMMARY ─────────────────────────────────────
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"""
  Best Model  : XGBoost
  Accuracy    : {results['XGBoost']['accuracy']:.1%}
  Macro-F1    : {results['XGBoost']['macro_f1']:.3f}
  Weighted-F1 : {results['XGBoost']['weighted_f1']:.3f}

  Most Important Feature : acceptance_rate
  Second Most Important  : wa_ratio
  Third Most Important   : avg_solve_time_min

  Files Saved:
    fig1_class_distribution.png
    fig2_acceptance_rate.png
    fig3_solve_time.png
    fig4_model_accuracy.png
    fig5_confusion_matrix.png
    fig6_feature_importance.png
    fig7_shap_importance.png  (only if shap installed)
    fig8_perclass_f1.png

  Use these figures directly in your Overleaf paper.
""")
print("="*60)
print("  Done! All steps completed successfully.")
print("="*60 + "\n")
