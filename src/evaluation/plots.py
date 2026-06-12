"""Plotting helpers for diagnosing the classical models (train vs test error).."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Consistent colors across all figures.
TRAIN_COLOR = "#1f77b4"  # blue
TEST_COLOR = "#d62728"   # red


def save_train_test_bars(
    labels,
    train_values,
    test_values,
    ylabel,
    title,
    out_path,
    annotate=True,
):
    """Grouped bar chart comparing a train metric vs a test metric per model.

    Args:
        labels: model names (x categories).
        train_values / test_values: one value per label.
        ylabel: y-axis label (e.g. "Error (1 − accuracy)" or "F1-score").
        title: figure title.
        out_path: where to write the PNG.
        annotate: write the numeric value on top of each bar.
    """
    labels = list(labels)
    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_train = ax.bar(
        x - width / 2,
        train_values,
        width,
        label="Entrenamiento",
        color=TRAIN_COLOR,
    )
    bars_test = ax.bar(
        x + width / 2,
        test_values,
        width,
        label="Prueba (test)",
        color=TEST_COLOR,
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    if annotate:
        for bars in (bars_train, bars_test):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overfit_gap_bar(labels, gaps, out_path, title):
    """Single-series bar chart of the train − CV F1 gap per model.

    A taller bar means the model fits its training data much better than the
    held-out folds, i.e. it overfits more.
    """
    labels = list(labels)
    gaps = np.asarray(gaps, dtype=float)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(x, gaps, width=0.6, color="#9467bd")

    ax.set_ylabel("Gap de sobreajuste  (F1 train − F1 CV)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_overfit_synthesis(labels, train_f1, cv_f1, out_path, title, ylim=None):
    """One-figure synthesis: per model, train F1 vs 5-fold CV F1.

    Reads the whole overfitting story at a glance: the height of the CV bar is
    the model's real quality, and the gap up to the training bar (annotated) is
    how much it overfits.
    """
    labels = list(labels)
    train_f1 = np.asarray(train_f1, dtype=float)
    cv_f1 = np.asarray(cv_f1, dtype=float)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.5))

    bars_tr = ax.bar(
        x - width / 2, train_f1, width,
        label="F1 entrenamiento", color=TRAIN_COLOR,
    )
    bars_cv = ax.bar(
        x + width / 2, cv_f1, width,
        label="F1 validación cruzada (5-fold)", color=TEST_COLOR,
    )

    for bars in (bars_tr, bars_cv):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    # Gap (overfitting) annotated above each pair, colored by severity:
    # <0.05 low (green), 0.05-0.10 moderate (orange), >0.10 high (red).
    def gap_color(gap):
        """Color the gap label by severity (green/orange/red)."""
        if gap < 0.05:
            return "#2ca02c"
        if gap <= 0.10:
            return "#ff7f0e"
        return "#d62728"

    for xi, (t, c) in zip(x, zip(train_f1, cv_f1)):
        gap = t - c
        ax.annotate(
            f"gap {gap:.3f}",
            xy=(xi, max(t, c)),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=gap_color(gap),
            fontweight="bold",
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel("F1-score (weighted)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_model_comparison(labels, means, stds, out_path, title, ylim=None):
    """Main comparison: mean 5-fold CV F1 per model (sorted best→worst).

    Error bars are ±1 standard deviation across folds. The best model is
    highlighted. This is the figure that answers "which model generalizes
    best to unseen data?".
    """
    labels = list(labels)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    order = np.argsort(means)[::-1]
    labels = [labels[i] for i in order]
    means = means[order]
    stds = stds[order]

    x = np.arange(len(labels))

    # Highlight the winner; everything else in a calmer blue.
    colors = ["#2ca02c" if i == 0 else "#4c72b0" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(
        x, means, yerr=stds, capsize=6, color=colors,
        error_kw={"ecolor": "#333", "elinewidth": 1.2},
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(
            f"{mean:.3f} ± {std:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean + std),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("F1-score (weighted) — validación cruzada 5-fold")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cv_vs_test(labels, cv_mean, cv_std, test_f1, out_path, title, ylim=None):
    """Closing-the-loop figure: CV F1 (the selection signal) vs external-test F1.

    Per model, the left bar is the mean 5-fold CV F1 (error bar = ±1 std) and
    the right bar is the F1 on the held-out external test set. If the test bar
    lands within/near the CV error bar, CV was a faithful predictor of unseen
    performance. Models are sorted by CV F1 (best→worst).
    """
    labels = list(labels)
    cv_mean = np.asarray(cv_mean, dtype=float)
    cv_std = np.asarray(cv_std, dtype=float)
    test_f1 = np.asarray(test_f1, dtype=float)

    order = np.argsort(cv_mean)[::-1]
    labels = [labels[i] for i in order]
    cv_mean = cv_mean[order]
    cv_std = cv_std[order]
    test_f1 = test_f1[order]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.5))

    bars_cv = ax.bar(
        x - width / 2, cv_mean, width, yerr=cv_std, capsize=6,
        color="#4c72b0", label="F1 validación cruzada (5-fold)",
        error_kw={"ecolor": "#333", "elinewidth": 1.2},
    )
    bars_te = ax.bar(
        x + width / 2, test_f1, width,
        color="#2ca02c", label="F1 test externo (375 tuits)",
    )

    for bars in (bars_cv, bars_te):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("F1-score (weighted)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_cv_test(
    mean_fpr,
    mean_tpr,
    std_tpr,
    cv_auc_mean,
    cv_auc_std,
    test_fpr,
    test_tpr,
    test_auc,
    out_path,
    title,
    pos_label="anorexia",
):
    """ROC curves: cross-validated (mean ±std band) vs external test.

    The CV curve is the mean of the per-fold ROC curves (band = ±1 std across
    folds); the test curve is the single ROC on the held-out test set. The
    dashed diagonal is the no-skill baseline. AUCs go in the legend.
    """
    mean_fpr = np.asarray(mean_fpr)
    mean_tpr = np.asarray(mean_tpr)
    std_tpr = np.asarray(std_tpr)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Azar (AUC = 0.5)")

    ax.plot(
        mean_fpr, mean_tpr, "-", color=TRAIN_COLOR, lw=2,
        label=f"Validación cruzada (AUC = {cv_auc_mean:.3f} ± {cv_auc_std:.3f})",
    )
    ax.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color=TRAIN_COLOR, alpha=0.15,
    )

    ax.plot(
        test_fpr, test_tpr, "-", color="#2ca02c", lw=2,
        label=f"Test externo (AUC = {test_auc:.3f})",
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel(f"Tasa de falsos positivos (clase positiva: {pos_label})")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curva ROC — {title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_auc_cv_vs_test(labels, cv_auc_mean, cv_auc_std, test_auc, out_path, title, ylim=None):
    """Grouped bars: CV AUC (mean ±std) vs external-test AUC, sorted best→worst."""
    labels = list(labels)
    cv_auc_mean = np.asarray(cv_auc_mean, dtype=float)
    cv_auc_std = np.asarray(cv_auc_std, dtype=float)
    test_auc = np.asarray(test_auc, dtype=float)

    order = np.argsort(cv_auc_mean)[::-1]
    labels = [labels[i] for i in order]
    cv_auc_mean = cv_auc_mean[order]
    cv_auc_std = cv_auc_std[order]
    test_auc = test_auc[order]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.bar(
        x - width / 2, cv_auc_mean, width, yerr=cv_auc_std, capsize=6,
        color="#4c72b0", label="AUC validación cruzada (5-fold)",
        error_kw={"ecolor": "#333", "elinewidth": 1.2},
    )
    ax.bar(
        x + width / 2, test_auc, width,
        color="#2ca02c", label="AUC test externo (375 tuits)",
    )

    for xi, cvv, tev in zip(x, cv_auc_mean, test_auc):
        ax.annotate(f"{cvv:.3f}", xy=(xi - width / 2, cvv), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{tev:.3f}", xy=(xi + width / 2, tev), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AUC (ROC)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
