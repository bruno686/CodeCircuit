import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

features = []
labels = []

with open("/home/kaiyu/CodeCircuit/data/gemma_mbpp_cpp_features.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        features.append(data["feature"])
        labels.append(data["step_labels"])

X = np.array(features)
y = np.array(labels)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

num_samples = len(X)
split_idx = int(num_samples * 0.7)  # 前 70%

X_train = X[:split_idx]
y_train = y[:split_idx]

X_test = X[split_idx:]
y_test = y[split_idx:]

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

print("\nDetailed classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

unique, counts = np.unique(y_test, return_counts=True)
ratio = counts / counts.sum()

print("Label counts in test set:")
for u, c, r in zip(unique, counts, ratio):
    print(f"Label {u}: {c} samples ({r:.2%})")

# 逐类准确率
acc_0 = np.mean(y_pred[y_test == 0] == 0)
acc_1 = np.mean(y_pred[y_test == 1] == 1)

print(f"Accuracy for label 0: {acc_0:.4f}")
print(f"Accuracy for label 1: {acc_1:.4f}")

from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------
# 针对 label 0 的 AUROC / AUPR / FPR@95
# -----------------------------

# 原模型输出是 y_prob = P(y=1)
# 对标签 0 来说，我们要的概率是 P(y=0) = 1 - P(y=1)
y_prob_0 = 1 - y_prob
y_true_0 = (y_test == 0).astype(int)  # 把 "是否为 0" 作为正类

# AUROC for label 0
auc_0 = roc_auc_score(y_true_0, y_prob_0)

# AUPR for label 0
aupr_0 = average_precision_score(y_true_0, y_prob_0)

# FPR@95 for label 0
# 找到 TPR = 95% 时的阈值（对 label 0）
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true_0, y_prob_0)

# 找到最接近 0.95 TPR 的索引
target_tpr = 0.95
idx = np.argmin(np.abs(tpr - target_tpr))
fpr_95 = fpr[idx]

print(f"\n=== Metrics for label 0 ===")
print(f"AUROC (label 0): {auc_0:.4f}")
print(f"AUPR  (label 0): {aupr_0:.4f}")
print(f"FPR@95(label 0): {fpr_95:.4f}")