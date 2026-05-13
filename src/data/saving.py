def save_dataframe(df, file_path):
    """Save dataframe as UTF-8 CSV."""
    df.to_csv(file_path, index=False, encoding="utf-8-sig")