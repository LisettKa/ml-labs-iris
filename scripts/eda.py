from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

# 1. Загрузка
df = pd.read_csv("data/raw/iris.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Dtypes:\n", df.dtypes)
print("Missing:\n", df.isna().sum())
print("Class balance:\n", df["species"].value_counts())

# 2. Сводки в CSV
out_dir = Path("reports/LAB1")
out_dir.mkdir(parents=True, exist_ok=True)
stats = df.describe(include="all").T
stats.to_csv(out_dir / "eda_stats.csv")
print(f"Saved {out_dir / 'eda_stats.csv'}")

# 3. Корреляции
corr = df.drop(columns=["target"]).corr(numeric_only=True)
corr.to_csv(out_dir / "correlations.csv")

# 4. Демонстрация проблемы варианта: переобучение
X = df.drop(columns=["target", "species"])
y = df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

depths = [1, 2, 3, 5, 10]
train_acc, test_acc = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_tr, y_tr)
    train_acc.append(clf.score(X_tr, y_tr))
    test_acc.append(clf.score(X_te, y_te))

# 5. График проблемы: переобучение при росте глубины
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(depths, train_acc, "o-", label="train accuracy")
axes[0].plot(depths, test_acc, "s-", label="test accuracy")
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("accuracy")
axes[0].set_title("Переобучение: train уходит в 1.0, test плато")
axes[0].legend()
axes[0].grid(True)

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    ax=axes[1],
)
axes[1].set_title("Petal length × width: классы почти линейно разделимы")

plt.tight_layout()
plt.savefig(out_dir / "problem_plot.png", dpi=150)
print(f"Saved {out_dir / 'problem_plot.png'}")

# 6. Числа для отчёта
print("\n=== ЧИСЛА ДЛЯ ОТЧЁТА ===")
print(f"n_rows = {len(df)}")
print(f"n_features = {X.shape[1]}")
for sp, cnt in df["species"].value_counts().items():
    print(f"class {sp}: {cnt} ({cnt/len(df)*100:.1f}%)")
print(
    f"corr(petal_length, petal_width) = {corr.loc['petal length (cm)', 'petal width (cm)']:.3f}"
)
for d, tr, te in zip(depths, train_acc, test_acc):
    print(f"depth={d}: train={tr:.3f} test={te:.3f}")
print(f"max train accuracy = {max(train_acc):.3f}")
print(f"best test accuracy = {max(test_acc):.3f}")
