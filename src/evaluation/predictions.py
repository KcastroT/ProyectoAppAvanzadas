import pandas as pd


def build_results_dataframe(
    X_test,
    y_test,
    y_pred,
):
    """Create dataframe with predictions."""

    return pd.DataFrame({
        "text": X_test.reset_index(drop=True),
        "real": y_test.reset_index(drop=True),
        "predicted": y_pred,
    })


def get_correct_predictions(results_df):
    """Return correct predictions."""

    return results_df[
        results_df["real"] == results_df["predicted"]
    ]


def get_incorrect_predictions(results_df):
    """Return incorrect predictions."""

    return results_df[
        results_df["real"] != results_df["predicted"]
    ]


def get_false_positives(
    results_df,
    positive_class,
):
    """Return false positives."""

    return results_df[
        (results_df["real"] != positive_class)
        & (results_df["predicted"] == positive_class)
    ]


def get_false_negatives(
    results_df,
    positive_class,
):
    """Return false negatives."""

    return results_df[
        (results_df["real"] == positive_class)
        & (results_df["predicted"] != positive_class)
    ]