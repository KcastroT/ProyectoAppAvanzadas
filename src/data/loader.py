import pandas as pd


def load_excel_dataset(file_path):
    """Load dataset from Excel."""
    return pd.read_excel(file_path)


def load_csv_dataset(file_path):
    """Load dataset from CSV."""
    return pd.read_csv(file_path)