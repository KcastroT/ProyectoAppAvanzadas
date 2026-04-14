import pandas as pd
import ftfy

df = pd.read_excel("data_train.xlsx")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)

df.to_csv("data_fixed_v2md .csv", index=False, encoding="utf-8-sig")