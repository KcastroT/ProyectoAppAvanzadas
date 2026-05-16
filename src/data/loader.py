import pandas as pd


def load_excel_dataset(file_path):
    """
    Load dataset from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(file_path)


def load_csv_dataset(file_path):
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file_path)