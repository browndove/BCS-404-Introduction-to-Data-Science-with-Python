"""
BCS 404 — Titanic analysis with a custom maritime chart theme.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Maritime palette -------------------------------------------------------
INK = "#0F1C2E"
SLATE = "#1E3A5F"
SEA = "#0E7C7B"
FOAM = "#3DCCC7"
BRASS = "#C4A35A"
CORAL = "#E07A5F"
MIST = "#E8EEF4"
PAPER = "#F5F7FA"
WHITE = "#FFFFFF"
GRID = "#C5D0DC"

SURVIVE = SEA
PERISH = CORAL
CLASS_COLORS = ["#1B4F72", "#2E86AB", "#A23B72"]  # 1st, 2nd, 3rd – distinct, not pastel junk

CMAP_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "atlantic", [CORAL, WHITE, SEA]
)
CMAP_HEAT = mcolors.LinearSegmentedColormap.from_list(
    "depth", ["#0F1C2E", "#1E3A5F", "#0E7C7B", "#3DCCC7", "#C4A35A"]
)
CMAP_MISSING = mcolors.LinearSegmentedColormap.from_list(
    "void", [PAPER, "#7EB8C9", "#1E3A5F"]
)


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": WHITE,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    sns.set_theme(style="whitegrid", rc=plt.rcParams)


def style_ax(
    ax,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    subtitle: str | None = None,
) -> None:
    """Lively left title, teal accent bar, optional italic subtitle."""
    ax.set_title(
        title, pad=26 if subtitle else 14, loc="left",
        color=INK, fontsize=15, fontweight="bold",
    )
    # Teal accent underline under the heading
    ax.plot(
        [0, 0.32], [1.0, 1.0], transform=ax.transAxes, clip_on=False,
        color=SEA, lw=3.2, solid_capstyle="round",
    )
    if subtitle:
        ax.text(
            0.0, 1.035, subtitle, transform=ax.transAxes, clip_on=False,
            ha="left", va="bottom", fontsize=9.5, color=SEA, style="italic",
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SLATE)
        ax.spines[spine].set_linewidth(1.1)
    ax.figure.subplots_adjust(top=0.86)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    clean = df.copy()
    clean["Age"] = clean["Age"].fillna(clean["Age"].median())
    clean["Embarked"] = clean["Embarked"].fillna(clean["Embarked"].mode()[0])
    clean = clean.drop(columns=["Cabin"]).drop_duplicates()
    return clean


def fig_missing(df_raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.heatmap(
        df_raw.isnull(), cbar=True, yticklabels=False, cmap=CMAP_MISSING,
        ax=ax, cbar_kws={"shrink": 0.6, "label": "Missing"},
    )
    style_ax(ax, "Where Is the Data Missing?", "Columns", "",
             subtitle="Bright cells mark incomplete passenger records")
    ax.set_yticks([])
    fig.savefig(FIG_DIR / "01_missing_values_heatmap.png")
    plt.close(fig)


def fig_age_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.histplot(
        df["Age"], bins=28, kde=True, color=SEA, edgecolor=WHITE,
        linewidth=0.8, alpha=0.88, ax=ax,
        line_kws={"color": BRASS, "lw": 2.6},
    )
    med, mean = df["Age"].median(), df["Age"].mean()
    ax.axvline(med, color=CORAL, ls="--", lw=2, label=f"Median  {med:.1f}")
    ax.axvline(mean, color=SLATE, ls=":", lw=2.2, label=f"Mean  {mean:.1f}")
    style_ax(ax, "How Old Were Titanic Passengers?", "Age (years)", "Count",
             subtitle="Most travellers were in their twenties and thirties")
    ax.legend(loc="upper right", fontsize=10)
    fig.text(
        0.99, 0.01, "Titanic · BCS 404", ha="right", va="bottom",
        color=SLATE, fontsize=8, alpha=0.5,
    )
    fig.savefig(FIG_DIR / "02_age_histogram.png")
    plt.close(fig)


def fig_pclass_bar(df: pd.DataFrame) -> None:
    counts = df["Pclass"].value_counts().sort_index()
    labels = ["1st Class\n(Upper)", "2nd Class\n(Middle)", "3rd Class\n(Lower)"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        labels, counts.values, color=CLASS_COLORS, width=0.62,
        edgecolor=WHITE, linewidth=1.5, zorder=3,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(val), ha="center", va="bottom", fontweight="bold",
            color=INK, fontsize=12,
        )
        # percentage inside
        pct = val / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
            f"{pct:.0f}%", ha="center", va="center", color=WHITE,
            fontweight="bold", fontsize=11,
        )
    style_ax(ax, "Who Sat Where? Class Breakdown", "", "Number of passengers",
             subtitle="Third class makes up more than half the ship")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.savefig(FIG_DIR / "03_pclass_barchart.png")
    plt.close(fig)


def fig_age_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    plot_df = df.copy()
    plot_df["Class"] = plot_df["Pclass"].map(
        {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
    )
    order = ["1st Class", "2nd Class", "3rd Class"]
    sns.violinplot(
        data=plot_df, x="Class", y="Age", hue="Class", order=order,
        palette=dict(zip(order, CLASS_COLORS)), legend=False,
        inner=None, cut=0, linewidth=0, saturation=0.9, ax=ax,
    )
    for coll in ax.collections:
        coll.set_alpha(0.32)
    sns.boxplot(
        data=plot_df, x="Class", y="Age", hue="Class", order=order,
        palette=dict(zip(order, CLASS_COLORS)), legend=False,
        width=0.22, showfliers=False, linewidth=1.4, ax=ax,
        boxprops={"zorder": 3}, whiskerprops={"zorder": 3},
        medianprops={"color": BRASS, "lw": 2.2, "zorder": 4},
        capprops={"zorder": 3},
    )
    sns.stripplot(
        data=plot_df.sample(min(280, len(plot_df)), random_state=42),
        x="Class", y="Age", order=order, color=INK, size=2.4, alpha=0.22,
        jitter=0.18, ax=ax, zorder=2,
    )
    style_ax(ax, "Younger in Steerage?", "Passenger class", "Age (years)",
             subtitle="Median age falls from 1st class down to 3rd")
    fig.savefig(FIG_DIR / "04_age_by_pclass_boxplot.png")
    plt.close(fig)


def fig_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for survived, color, label, marker in [
        (0, PERISH, "Did not survive", "o"),
        (1, SURVIVE, "Survived", "D"),
    ]:
        sub = df[df["Survived"] == survived]
        ax.scatter(
            sub["Age"], sub["Fare"], c=color, label=label, marker=marker,
            s=36, alpha=0.72, edgecolors=WHITE, linewidths=0.45, zorder=3,
        )
    style_ax(ax, "Did Higher Fares Mean Better Odds?", "Age (years)", "Fare",
             subtitle="Teal diamonds survived · coral circles did not")
    ax.set_ylim(-5, min(df["Fare"].max() * 1.05, 320))
    ax.legend(loc="upper right", markerscale=1.3, fontsize=10)
    # annotate extreme fare
    top = df.loc[df["Fare"].idxmax()]
    if top["Fare"] > 200:
        ax.annotate(
            f"Fare {top['Fare']:.0f}",
            xy=(top["Age"], top["Fare"]),
            xytext=(top["Age"] + 8, min(top["Fare"], 300) - 40),
            fontsize=8, color=SLATE,
            arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8),
        )
    fig.savefig(FIG_DIR / "05_age_vs_fare_scatter.png")
    plt.close(fig)


def fig_corr(df: pd.DataFrame) -> None:
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "PassengerId"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=CMAP_DIVERGING,
        center=0, square=True, linewidths=1.2, linecolor=PAPER,
        cbar_kws={"shrink": 0.72, "label": "Pearson r"},
        annot_kws={"size": 9, "weight": "medium"},
        vmin=-0.7, vmax=0.7, ax=ax,
    )
    style_ax(ax, "What Moves Together?", "", "",
             subtitle="Pearson correlations among numerical variables")
    ax.grid(False)
    fig.savefig(FIG_DIR / "06_correlation_heatmap.png")
    plt.close(fig)


def fig_pairplot(df: pd.DataFrame) -> None:
    pair_vars = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    g = sns.pairplot(
        df[pair_vars],
        hue="Survived",
        palette={0: PERISH, 1: SURVIVE},
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.55, "s": 28, "edgecolor": WHITE, "linewidth": 0.3},
        diag_kws={"fill": True, "alpha": 0.45, "linewidth": 1.5},
    )
    g.fig.patch.set_facecolor(PAPER)
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_facecolor(WHITE)
            ax.grid(True, linestyle="--", alpha=0.4)
    g._legend.set_title("Survived")
    g.fig.suptitle("Survival Patterns at a Glance", y=1.03,
                    fontsize=15, fontweight="bold", color=INK, ha="left", x=0.08)
    g.fig.text(0.08, 1.005, "Pairwise views of class, age, family size and fare",
               fontsize=9, color=SEA, style="italic", ha="left")
    g.savefig(FIG_DIR / "07_pairplot.png")
    plt.close("all")


def fig_confusion(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.8))
    labels = ["Did not\nsurvive", "Survived"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=mcolors.LinearSegmentedColormap.from_list(
            "cm", [PAPER, "#A8DADC", SEA, "#064E4D"]
        ),
        xticklabels=labels, yticklabels=labels,
        linewidths=2, linecolor=PAPER, square=True, cbar=False,
        annot_kws={"size": 22, "weight": "bold", "color": INK},
        ax=ax,
    )
    # percentage annotations under counts
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, color=SLATE)
    style_ax(ax, "Did the Model Call It Right?", "Predicted", "Actual",
             subtitle="Logistic Regression on the held-out test set")
    ax.grid(False)
    acc = (cm[0, 0] + cm[1, 1]) / total
    ax.text(
        1.0, -0.18, f"Accuracy  {acc:.1%}", transform=ax.transAxes,
        ha="right", fontsize=11, fontweight="bold", color=SEA,
    )
    fig.savefig(FIG_DIR / "08_confusion_matrix.png")
    plt.close(fig)


def train_and_cm(df: pd.DataFrame) -> None:
    ml = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    ml["Sex"] = LabelEncoder().fit_transform(ml["Sex"])
    ml["Embarked"] = LabelEncoder().fit_transform(ml["Embarked"])
    X, y = ml.drop(columns=["Survived"]), ml["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    fig_confusion(cm)


def main() -> None:
    apply_theme()
    raw = pd.read_csv(DATA_PATH)
    df = load_and_clean(DATA_PATH)
    print("Generating styled figures…")
    fig_missing(raw)
    fig_age_hist(df)
    fig_pclass_bar(df)
    fig_age_box(df)
    fig_scatter(df)
    fig_corr(df)
    fig_pairplot(df)
    train_and_cm(df)
    print(f"Saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
