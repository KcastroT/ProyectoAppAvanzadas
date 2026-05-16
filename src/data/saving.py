def save_dataframe(df, file_path):
    """Save a pandas DataFrame to a CSV file using UTF-8 encoding with BOM.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be saved.
    file_path : str or pathlib.Path
        Destination path for the output CSV file.
    """
    df.to_csv(file_path, index=False, encoding="utf-8-sig")