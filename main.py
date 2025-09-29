import re
import pandas as pd
import os

base = "hf://datasets/trl-lib/tldr/"
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

# Cleaner: behält SUBREDDIT-Zeile, entfernt TITLE, TL;DR, '---', normalisiert Whitespace
def clean_reddit_keep_subreddit(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # SUBREDDIT-Zeile sichern (ganze Zeile behalten)
    m_sub = re.search(r'^\s*SUBREDDIT:\s*.+\s*$', text, flags=re.I | re.M)
    subreddit_line = m_sub.group(0).strip() if m_sub else ""

    # POST-Body extrahieren: nach "POST:" bis "TL;DR" oder Ende
    body = text
    parts_after_post = re.split(r'(?i)\bPOST:\s*', body, maxsplit=1)
    body = parts_after_post[1] if len(parts_after_post) == 2 else body
    body = re.split(r'(?i)\bTL;DR\b\s*:?', body, maxsplit=1)[0]

    # offensichtliche Metadaten/Markdown raus
    body = re.sub(r'(?m)^\s*TITLE:.*$', '', body)     # Titelzeile entfernen, falls noch vorhanden
    body = re.sub(r'-{3,}', ' ', body)                # '---' Linien
    body = re.sub(r'\s+', ' ', body).strip()          # Whitespace normalisieren

    # Ergebnis zusammenbauen
    if subreddit_line:
        return f"{subreddit_line}\n\n{body}".strip()
    return body

def load_and_merge(path,
                   prompt_col="prompt",
                   completion_col="completion",
                   keep_original=False,
                   sep="\n---\n",
                   clean_prompt=True):
    # nur nötige Spalten laden
    df = pd.read_parquet(base + path, columns=[prompt_col, completion_col]).copy()

    # prompt bereinigen (SUBREDDIT drin lassen, TL;DR/TITLE usw. entfernen)
    if clean_prompt:
        df[prompt_col] = df[prompt_col].apply(clean_reddit_keep_subreddit)
    else:
        # falls nicht bereinigt wird, zumindest NaNs -> ""
        df[prompt_col] = df[prompt_col].fillna("").astype(str)

    # completion säubern nur bzgl. NaN/Typ
    df[completion_col] = df[completion_col].fillna("").astype(str)

    # zusammenführen zu einer Spalte "text"
    df["text"] = df[prompt_col] + sep + df[completion_col]

    if not keep_original:
        df = df.drop(columns=[prompt_col, completion_col])

    return df

# Hilfsfunktion zum Filtern nach Subreddit
def filter_subreddits(df, subreddits=("r/relationships", "r/dating_advice")):
    # Pattern mit r/... bauen
    pattern = "|".join([fr"SUBREDDIT:\s*{sr}\b" for sr in subreddits])
    return df[df["text"].str.contains(pattern, case=False, regex=True)].reset_index(drop=True)

# Lade + merge + filter
df_train = filter_subreddits(load_and_merge(splits["train"]))
df_val   = filter_subreddits(load_and_merge(splits["validation"]))
df_test  = filter_subreddits(load_and_merge(splits["test"]))

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# Konsolen-Annotation-Tool
def annotate_texts(df, output_file="annotations.csv", limit=None):
    annotations = []
    toxic_count = 0
    non_toxic_count = 0

    # ggf. nur N Texte
    if limit is not None:
        df = df.head(limit)
    total = len(df)

    for idx, row in df.iterrows():
        text = row["text"]

        # Text anzeigen
        print("=" * 80)
        print(f"ID: {idx}")
        print(text)
        print("=" * 80)
        print("[t] toxisch  |  [n] nicht toxisch  |  [q] beenden")

        while True:
            choice = input("Eingabe: ").strip().lower()
            if choice in ["t", "n", "q"]:
                break
            else:
                print("Ungültige Eingabe, bitte 't', 'n' oder 'q' eingeben.")

        if choice == "q":
            print("Abbruch durch Benutzer.")
            break
        elif choice == "t":
            label = "toxic"
            toxic_count += 1
        else:
            label = "non-toxic"
            non_toxic_count += 1

        annotations.append({"id": idx, "text": text, "label": label})

        # Fortschritt + Zähler
        done = len(annotations)
        print(f"👉 Aktueller Stand: toxisch = {toxic_count} | nicht toxisch = {non_toxic_count}  |  Fortschritt: {done}/{total}")
        print("-" * 80)

    # Ergebnisse speichern
    out_df = pd.DataFrame(annotations)
    out_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Annotationen gespeichert in: {os.path.abspath(output_file)}")

# Tool starten
#if __name__ == "__main__":
#    annotate_texts(df_test, output_file="df_test_annotations.csv", limit=1500)

# Anzahl der Beispiele für Val und Train
VAL_SIZE = 600
TRAIN_SIZE = 4800

# Validation vorbereiten
df_val_sample = df_val.sample(n=VAL_SIZE, random_state=42).reset_index(drop=True)
df_val_out = pd.DataFrame({
    "id": df_val_sample.index,
    "text": df_val_sample["text"],
   "label": [""] * len(df_val_sample)  # leere Labels
})
df_val_out.to_csv("df_val_annotations.csv", index=False, encoding="utf-8")
print(f"✅ df_val_annotations.csv gespeichert mit {len(df_val_out)} Einträgen.")

# Train vorbereiten
df_train_sample = df_train.iloc[10000:20000].reset_index(drop=True)
df_train_out = pd.DataFrame({
    "id": df_train_sample.index,
    "text": df_train_sample["text"],
    "label": [""] * len(df_train_sample)
})
df_train_out.to_csv("df_train_annotations.csv", index=False, encoding="utf-8")
print(f"✅ df_train_annotations.csv gespeichert mit {len(df_train_out)} Einträgen.")

