"""Plotting helpers for diagnosing the classical models (train vs test error).

Pure plotting: callers compute the numbers and pass arrays in. Uses the
non-interactive ``Agg`` backend so figures are written straight to disk
without needing a display.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Consistent colors across all figures.
TRAIN_COLOR = "#1f77b4"  # blue
TEST_COLOR = "#d62728"   # red


def save_learning_curve(
    train_sizes,
    train_error_mean,
    train_error_std,
    test_error_mean,
    test_error_std,
    out_path,
    title,
):
    """Plot training vs test error as a function of the training-set size.

    A large persistent gap (low train error, high test error) signals
    overfitting; both errors high and close together signals underfitting.

    Args:
        train_sizes: 1-D array of absolute training-set sizes (x axis).
        train_error_mean / train_error_std: error on the training subset.
        test_error_mean / test_error_std: error on the external test set.
        out_path: where to write the PNG.
        title: figure title (usually the model name).
    """
    train_sizes = np.asarray(train_sizes)
    train_error_mean = np.asarray(train_error_mean)
    train_error_std = np.asarray(train_error_std)
    test_error_mean = np.asarray(test_error_mean)
    test_error_std = np.asarray(test_error_std)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        train_sizes,
        train_error_mean,
        "o-",
        color=TRAIN_COLOR,
        label="Error de entrenamiento",
    )
    ax.fill_between(
        train_sizes,
        train_error_mean - train_error_std,
        train_error_mean + train_error_std,
        alpha=0.15,
        color=TRAIN_COLOR,
    )

    ax.plot(
        train_sizes,
        test_error_mean,
        "s-",
        color=TEST_COLOR,
        label="Error de prueba (test)",
    )
    ax.fill_between(
        train_sizes,
        test_error_mean - test_error_std,
        test_error_mean + test_error_std,
        alpha=0.15,
        color=TEST_COLOR,
    )

    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Error de clasificación (1 − accuracy)")
    ax.set_title(f"Curva de aprendizaje — {title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_learning_curve_dual(
    train_sizes,
    train_error_mean,
    test_error_mean,
    out_path,
    title,
    error_ylim=None,
    acc_ylim=None,
):
    """Keras-style two-panel learning curve (error + accuracy).

    Mirrors the familiar "Training and validation loss / accuracy" layout:
    the training metric is drawn as dots and the test metric as a solid line,
    both in blue. Pass shared ``error_ylim`` / ``acc_ylim`` so every model's
    figure uses the same scale and can be compared side by side.

    Args:
        train_sizes: 1-D array of absolute training-set sizes (x axis).
        train_error_mean: error on the training subset (1 − accuracy).
        test_error_mean: error on the external test set (1 − accuracy).
        out_path: where to write the PNG.
        title: overall figure title (the model name).
        error_ylim / acc_ylim: (low, high) tuples shared across all models.
    """
    train_sizes = np.asarray(train_sizes)
    train_error_mean = np.asarray(train_error_mean)
    test_error_mean = np.asarray(test_error_mean)

    train_acc = 1.0 - train_error_mean
    test_acc = 1.0 - test_error_mean

    fig, (ax_err, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    # ----- left: error ("loss" analog) -----
    ax_err.plot(
        train_sizes, train_error_mean, "o", color=TRAIN_COLOR,
        label="Error de entrenamiento",
    )
    ax_err.plot(
        train_sizes, test_error_mean, "-", color=TRAIN_COLOR,
        label="Error de prueba (test)",
    )
    ax_err.set_title("Error de entrenamiento y prueba")
    ax_err.set_xlabel("Número de ejemplos de entrenamiento")
    ax_err.set_ylabel("Error (1 − accuracy)")
    if error_ylim is not None:
        ax_err.set_ylim(error_ylim)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    # ----- right: accuracy -----
    ax_acc.plot(
        train_sizes, train_acc, "o", color=TRAIN_COLOR,
        label="Accuracy de entrenamiento",
    )
    ax_acc.plot(
        train_sizes, test_acc, "-", color=TRAIN_COLOR,
        label="Accuracy de prueba (test)",
    )
    ax_acc.set_title("Accuracy de entrenamiento y prueba")
    ax_acc.set_xlabel("Número de ejemplos de entrenamiento")
    ax_acc.set_ylabel("Accuracy")
    if acc_ylim is not None:
        ax_acc.set_ylim(acc_ylim)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="best")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


def save_validation_curve(
    x_values,
    train_mean,
    train_std,
    cv_mean,
    cv_std,
    out_path,
    title,
    xlabel,
    log_x=False,
    x_tick_labels=None,
    best_idx=None,
    ylim=None,
):
    """Plot train vs cross-validation score as a complexity knob is swept.

    Overfitting reads directly off this curve: the training score climbs
    toward 1.0 while the cross-validation score plateaus or drops, and the
    vertical gap between the two is the amount of overfitting.

    Args:
        x_values: numeric x positions (the swept parameter, or 0..n-1 indices
            when the parameter is categorical such as ``max_depth`` with None).
        train_mean / train_std: training F1 across CV folds.
        cv_mean / cv_std: validation (held-out fold) F1.
        out_path: where to write the PNG.
        title: figure title (the model name).
        xlabel: x-axis label.
        log_x: use a logarithmic x-axis (for a C sweep).
        x_tick_labels: explicit tick labels (for categorical sweeps).
        best_idx: index of the best-CV point to mark with a dashed line.
        ylim: shared (low, high) y-limits across models.
    """
    x_values = np.asarray(x_values, dtype=float)
    train_mean = np.asarray(train_mean)
    train_std = np.asarray(train_std)
    cv_mean = np.asarray(cv_mean)
    cv_std = np.asarray(cv_std)

    fig, ax = plt.subplots(figsize=(8, 5))

    if log_x:
        ax.set_xscale("log")

    ax.plot(
        x_values, train_mean, "o-", color=TRAIN_COLOR,
        label="F1 entrenamiento",
    )
    ax.fill_between(
        x_values, train_mean - train_std, train_mean + train_std,
        alpha=0.15, color=TRAIN_COLOR,
    )

    ax.plot(
        x_values, cv_mean, "s-", color=TEST_COLOR,
        label="F1 validación cruzada (5-fold)",
    )
    ax.fill_between(
        x_values, cv_mean - cv_std, cv_mean + cv_std,
        alpha=0.15, color=TEST_COLOR,
    )

    if best_idx is not None:
        ax.axvline(x_values[best_idx], ls="--", color="gray", alpha=0.7)
        ax.annotate(
            "mejor CV",
            xy=(x_values[best_idx], cv_mean[best_idx]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            color="gray",
            fontsize=9,
        )

    if x_tick_labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_tick_labels)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1-score (weighted)")
    ax.set_title(f"Curva de validación — {title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

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


def save_learning_curve_cv(
    train_sizes,
    train_mean,
    train_std,
    val_mean,
    val_std,
    out_path,
    title,
    ylim=None,
):
    """Learning curve from stratified 5-fold CV (F1, train vs validation).

    For each training-set size, the training and validation F1 are averaged
    across the CV folds; the shaded band is ±1 standard deviation. Reading it:
    both curves low and close = underfitting (high bias); a wide persistent gap
    = overfitting (high variance); a still-rising validation curve = more data
    would help.

    Args:
        train_sizes: absolute training-set sizes (x axis).
        train_mean / train_std: training F1 mean and std across folds.
        val_mean / val_std: validation (held-out fold) F1 mean and std.
        out_path: where to write the PNG.
        title: figure title (the model name).
        ylim: shared (low, high) y-limits across models.
    """
    train_sizes = np.asarray(train_sizes)
    train_mean = np.asarray(train_mean)
    train_std = np.asarray(train_std)
    val_mean = np.asarray(val_mean)
    val_std = np.asarray(val_std)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        train_sizes, train_mean, "o-", color=TRAIN_COLOR,
        label="F1 entrenamiento",
    )
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std,
        alpha=0.15, color=TRAIN_COLOR,
    )

    ax.plot(
        train_sizes, val_mean, "o-", color=TEST_COLOR,
        label="F1 validación (CV 5-fold)",
    )
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std,
        alpha=0.15, color=TEST_COLOR,
    )

    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("F1-score (weighted)")
    ax.set_title(f"Curva de aprendizaje — {title}")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

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
