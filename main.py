import pandas as pd

df = pd.read_excel("data_train.xlsx")

BAD_MARKERS = ("Ã", "Â", "â", "\x91", "\x92", "\x93", "\x94", "\x96", "\x97", "\x85")

def looks_broken(s: str) -> bool:
    return any(m in s for m in BAD_MARKERS)

def try_fix_once(s: str) -> str:
    candidates = [s]

    for enc in ("latin1", "cp1252"):
        try:
            candidates.append(s.encode(enc).decode("utf-8"))
        except:
            pass

    best = min(candidates, key=lambda x: sum(x.count(m) for m in BAD_MARKERS))
    return best

def fix_text(s):
    if not isinstance(s, str):
        return s

    prev = s
    for _ in range(3):  # varias pasadas por si viene doblemente roto
        new = try_fix_once(prev)
        if new == prev:
            break
        prev = new

    replacements = {
        "â€¦": "…",
        "â€œ": "“",
        "â€\x9d": "”",
        "â€˜": "‘",
        "â€™": "’",
        "â€“": "–",
        "â€”": "—",
        "â™¡": "♡",
        "Â¡": "¡",
        "Â¿": "¿",
        "Â«": "«",
        "Â»": "»",
        "Â ": " ",
        "ï¬\x81": "fi",
        "ï¬\x82": "fl",
    }

    for bad, good in replacements.items():
        prev = prev.replace(bad, good)

    return prev

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(fix_text)

df.to_csv("data_fixed.csv", index=False, encoding="utf-8-sig")