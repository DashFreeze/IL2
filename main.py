import re
import os
import random
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
try:
    from datasets import Dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

base = "hf://datasets/trl-lib/tldr/"
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

# Beim Annotationstool und speziellen Funktionen hat ChatGPT geholfen:
# behält Subreddit-Zeile, entfernt TITLE, TL;DR, '---', normalisiert Whitespace
def clean_reddit_keep_subreddit(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    m_sub = re.search(r'^\s*SUBREDDIT:\s*.+\s*$', text, flags=re.I | re.M)
    subreddit_line = m_sub.group(0).strip() if m_sub else ""
    body = text
    parts_after_post = re.split(r'(?i)\bPOST:\s*', body, maxsplit=1)
    body = parts_after_post[1] if len(parts_after_post) == 2 else body
    body = re.split(r'(?i)\bTL;DR\b\s*:?', body, maxsplit=1)[0]
    body = re.sub(r'(?m)^\s*TITLE:.*$', '', body)
    body = re.sub(r'-{3,}', ' ', body)
    body = re.sub(r'\s+', ' ', body).strip()
    if subreddit_line:
        return f"{subreddit_line}\n\n{body}".strip()
    return body

def load_and_merge(path,
                   prompt_col="prompt",
                   completion_col="completion",
                   keep_original=False,
                   sep="\n---\n",
                   clean_prompt=True):
    df = pd.read_parquet(base + path, columns=[prompt_col, completion_col]).copy()
    if clean_prompt:
        df[prompt_col] = df[prompt_col].apply(clean_reddit_keep_subreddit)
    else:
        df[prompt_col] = df[prompt_col].fillna("").astype(str)
    df[completion_col] = df[completion_col].fillna("").astype(str)
    df["text"] = df[prompt_col] + sep + df[completion_col]
    if not keep_original:
        df = df.drop(columns=[prompt_col, completion_col])
    return df

def filter_subreddits(df, subreddits=("r/relationships", "r/dating_advice")):
    pattern = "|".join([fr"SUBREDDIT:\s*{sr}\b" for sr in subreddits])
    return df[df["text"].str.contains(pattern, case=False, regex=True)].reset_index(drop=True)

RAW_DF_TRAIN = filter_subreddits(load_and_merge(splits["train"]))
RAW_DF_VAL   = filter_subreddits(load_and_merge(splits["validation"]))
RAW_DF_TEST  = filter_subreddits(load_and_merge(splits["test"]))

print(f"Train: {len(RAW_DF_TRAIN)} | Val: {len(RAW_DF_VAL)} | Test: {len(RAW_DF_TEST)}")


# Annotation

def annotate_texts(df, output_file="annotations.csv", limit=None):
    annotations = []
    toxic_count = 0
    non_toxic_count = 0
    if limit is not None:
        df = df.head(limit)
    total = len(df)
    for idx, row in df.iterrows():
        text = row["text"]
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
        done = len(annotations)
        print(f"👉 Aktueller Stand: toxisch = {toxic_count} | nicht toxisch = {non_toxic_count}  |  Fortschritt: {done}/{total}")
        print("-" * 80)
    out_df = pd.DataFrame(annotations)
    out_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Annotationen gespeichert in: {os.path.abspath(output_file)}")

VAL_SIZE = 600
TRAIN_SIZE = 4800

df_val_sample = RAW_DF_VAL.sample(n=VAL_SIZE, random_state=42).reset_index(drop=True)
df_val_out = pd.DataFrame({
    "id": df_val_sample.index,
    "text": df_val_sample["text"],
   "label": [""] * len(df_val_sample)
})
"""
df_val_out.to_csv("df_val_annotations.csv", index=False, encoding="utf-8")
print(f"✅ df_val_annotations.csv gespeichert mit {len(df_val_out)} Einträgen.")
"""

df_train_sample = RAW_DF_TRAIN.iloc[10000:20000].reset_index(drop=True)
df_train_out = pd.DataFrame({
    "id": df_train_sample.index,
    "text": df_train_sample["text"],
    "label": [""] * len(df_train_sample)
})
"""
df_train_out.to_csv("df_train_annotations.csv", index=False, encoding="utf-8")
print(f"✅ df_train_annotations.csv gespeichert mit {len(df_train_out)} Einträgen.")
"""

"""
# Damit wurden die Annotationen durchgeführt
if __name__ == "__main__":
    annotate_texts(df_test)
"""

# Preprocessing

LABEL_MAP = {
    "toxic": 1,
    "non-toxic": 0,
}

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    missing = [c for c in ["text", "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in {path}: {missing} (erwartet: text, label)")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if not set(df["label"]).issubset(set(k.lower() for k in LABEL_MAP.keys())):
        unknown = sorted(set(df["label"]) - set(k.lower() for k in LABEL_MAP.keys()))
        raise ValueError(f"Unbekannte Label-Werte in {path}: {unknown}. "
                         f"Passe LABEL_MAP in data_loading.py an.")
    df["y"] = df["label"].map(lambda s: LABEL_MAP[s])
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    return df[["text", "y"]]

def _class_stats(df: pd.DataFrame, name: str):
    vc = df["y"].value_counts().sort_index()
    total = len(df)
    print(f"— {name}: {total} Beispiele")
    for cls in [0, 1]:
        n = int(vc.get(cls, 0))
        pct = (n / total * 100) if total else 0.0
        print(f"   Klasse {cls} ({'non-toxic' if cls==0 else 'toxic'}): {n} ({pct:.1f}%)")

def _to_numpy(df: pd.DataFrame):
    X = df["text"].to_numpy()
    y = df["y"].to_numpy().astype(int)
    return X, y

def _to_hf_dataset(df: pd.DataFrame):
    if not HF_AVAILABLE:
        raise ImportError("`datasets` ist nicht installiert. Installiere mit: pip install datasets")
    return Dataset.from_pandas(df.rename(columns={"y": "label"}), preserve_index=False)

def load_splits(
    data_dir: str = "data",
    return_hf: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    paths = {
        "train": os.path.join(data_dir, "train.csv"),
        "eval":  os.path.join(data_dir, "eval.csv"),
        "test":  os.path.join(data_dir, "test.csv"),
    }
    train_df = _read_csv(paths["train"])
    eval_df  = _read_csv(paths["eval"])
    test_df  = _read_csv(paths["test"])
    if limit is not None:
        train_df = train_df.sample(n=min(limit, len(train_df)), random_state=SEED).reset_index(drop=True)
        eval_df  = eval_df.sample(n=min(limit, len(eval_df)), random_state=SEED).reset_index(drop=True)
        test_df  = test_df.sample(n=min(limit, len(test_df)), random_state=SEED).reset_index(drop=True)
    print("Klassenverteilungen:")
    _class_stats(train_df, "Train")
    _class_stats(eval_df,  "Eval")
    _class_stats(test_df,  "Test")
    if return_hf:
        train_ds = _to_hf_dataset(train_df)
        eval_ds  = _to_hf_dataset(eval_df)
        test_ds  = _to_hf_dataset(test_df)
        return {"hf": {"train": train_ds, "eval": eval_ds, "test": test_ds}}
    else:
        Xtr, ytr = _to_numpy(train_df)
        Xev, yev = _to_numpy(eval_df)
        Xte, yte = _to_numpy(test_df)
        return {"np": {"X_train": Xtr, "y_train": ytr,
                       "X_eval": Xev,  "y_eval": yev,
                       "X_test": Xte,  "y_test": yte}}


# Reproduzierbarkeit
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# entfernt die Kopfzeile "SUBREDDIT: ..." – NACHDEM du schon nach Subreddits gefiltert hast
SUBREDDIT_HEADER_RE = re.compile(r'(?mi)^\s*SUBREDDIT:\s*\S+\s*$')

def strip_subreddit_header(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = SUBREDDIT_HEADER_RE.sub("", text)
    return re.sub(r'\n{2,}', '\n', text).strip()

# spaCy nur für Baseline (TF-IDF)
def _build_spacy_en():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except Exception as e:
        print(f"[Lemma] spaCy/en_core_web_sm nicht verfügbar ({e}). Fahre ohne Lemmatisierung fort.")
        return None

def lemmatize_text_en(text: str, nlp) -> str:
    if not nlp:
        return text
    doc = nlp(text)
    return " ".join(t.lemma_ for t in doc if not t.is_punct and not t.is_space)

def preprocess_baseline_texts(texts: np.ndarray, nlp=None) -> np.ndarray:
    out = []
    for t in texts:
        t2 = strip_subreddit_header(t)
        t2 = lemmatize_text_en(t2, nlp)
        out.append(t2)
    return np.array(out, dtype=object)

# Baseline (TF-IDF mit Logistic Regression)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

splits_np = load_splits(data_dir="data", return_hf=False)["np"]
Xtr, ytr = splits_np["X_train"], splits_np["y_train"]
Xev, yev = splits_np["X_eval"],  splits_np["y_eval"]
Xte, yte = splits_np["X_test"],  splits_np["y_test"]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

_nlp = _build_spacy_en()
Xtr_prep = preprocess_baseline_texts(Xtr, _nlp)
Xev_prep = preprocess_baseline_texts(Xev, _nlp)
Xte_prep = preprocess_baseline_texts(Xte, _nlp)

pipe.fit(Xtr_prep, ytr)
print("== EVAL ==")
print(classification_report(yev, pipe.predict(Xev_prep), digits=3))
print("== TEST ==")
print(classification_report(yte, pipe.predict(Xte_prep), digits=3))

# BERT
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch

MODEL_NAME = "distilbert-base-uncased"

ds = load_splits(data_dir="data", return_hf=True)["hf"]
train_ds, eval_ds, test_ds = ds["train"], ds["eval"], ds["test"]

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# Für BERT nur den SUBREDDIT-Header entfernen 
def tokenize(batch):
    cleaned = [strip_subreddit_header(t) for t in batch["text"]]
    return tok(cleaned, truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds  = eval_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ["input_ids","attention_mask","label"]])
eval_ds  = eval_ds.remove_columns([c for c in eval_ds.column_names  if c not in ["input_ids","attention_mask","label"]])
test_ds  = test_ds.remove_columns([c for c in test_ds.column_names  if c not in ["input_ids","attention_mask","label"]])
train_ds.set_format("torch")
eval_ds.set_format("torch")
test_ds.set_format("torch")

labels_np = np.array(train_ds["label"])
class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=labels_np)
class_weights = torch.tensor(class_weights, dtype=torch.float)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, problem_type="single_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


fp16 = torch.cuda.is_available()

args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=42,
    logging_steps=50,
    fp16=fp16,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,  # statt tokenizer=tok
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


trainer.train()
print("== EVAL ==")
print(trainer.evaluate(eval_dataset=eval_ds))
print("== TEST ==")
print(trainer.evaluate(eval_dataset=test_ds))


# Vergleich: Baseline vs. BERT
from sklearn.metrics import confusion_matrix

def summarize(split_name, model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "split": split_name,
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(p, 4),
        "recall_macro": round(r, 4),
        "f1_macro": round(f1, 4),
    }

y_pred_ev_base = pipe.predict(Xev_prep)
y_pred_te_base = pipe.predict(Xte_prep)

pred_ev_bert = trainer.predict(eval_ds)
y_pred_ev_bert = np.argmax(pred_ev_bert.predictions, axis=-1)

pred_te_bert = trainer.predict(test_ds)
y_pred_te_bert = np.argmax(pred_te_bert.predictions, axis=-1)

rows = []
rows.append(summarize("eval",  "baseline_tfidf_logreg", yev, y_pred_ev_base))
rows.append(summarize("eval",  "bert_distilbert",       eval_ds["label"], y_pred_ev_bert))
rows.append(summarize("test",  "baseline_tfidf_logreg", yte, y_pred_te_base))
rows.append(summarize("test",  "bert_distilbert",       test_ds["label"], y_pred_te_bert))

df_cmp = pd.DataFrame(rows).sort_values(["split","f1_macro"], ascending=[True, False])
print("\n=== Vergleich Baseline vs. BERT ===")
print(df_cmp.to_string(index=False))

df_cmp.to_csv("metrics_summary.csv", index=False, encoding="utf-8")
print("\nGespeichert: metrics_summary.csv")

def show_cm(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print(f"\nConfusion Matrix [{name}] (rows=true, cols=pred)  labels=[0,1]:\n{cm}")

show_cm("Eval - Baseline", yev, y_pred_ev_base)
show_cm("Eval - BERT",     eval_ds["label"], y_pred_ev_bert)
show_cm("Test - Baseline", yte, y_pred_te_base)
show_cm("Test - BERT",     test_ds["label"], y_pred_te_bert)

#Cross-Validation: Baseline (TF-IDF + LogisticRegression)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def run_baseline_cv(data_dir="data", k=5, repeats=1, seed=42):
    print("\n[CV|Baseline] Lade train.csv für Cross-Validation …")
    train_df = _read_csv(os.path.join(data_dir, "train.csv"))
    X = train_df["text"].to_numpy()
    y = train_df["y"].to_numpy().astype(int)
    print(f"[CV|Baseline] Train-Set: {len(X)} Beispiele | Klassenverteilung: "
          f"{(y==0).sum()} non-toxic, {(y==1).sum()} toxic")

    _nlp_cv = _build_spacy_en()

    # Repeated Stratified K-Fold
    if repeats and repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
        total_folds = k * repeats
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        total_folds = k

    rows = []
    for global_fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        run_idx = 1 if repeats <= 1 else int(np.ceil(global_fold / k))
        fold_in_run = ((global_fold - 1) % k) + 1

        print(f"\n[CV|Baseline] === Run {run_idx}/{repeats if repeats>1 else 1} | Fold {fold_in_run}/{k} "
              f"(global {global_fold}/{total_folds}) ===")
        print(f"[CV|Baseline] Train-Indices: {len(tr_idx)} | Val-Indices: {len(va_idx)}")

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr = preprocess_baseline_texts(X_tr, _nlp_cv)
        X_va = preprocess_baseline_texts(X_va, _nlp_cv)

        clf = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])

        print("[CV|Baseline] Fitte Modell auf Fold-Train …")
        clf.fit(X_tr, y_tr)

        print("[CV|Baseline] Evaluiere auf Fold-Validierung …")
        y_pred = clf.predict(X_va)

        acc = accuracy_score(y_va, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_va, y_pred, average="macro", zero_division=0)
        print(f"[CV|Baseline] Fold {fold_in_run} -> acc={acc:.4f} p={p:.4f} r={r:.4f} f1={f1:.4f}")
        rows.append((run_idx, fold_in_run, acc, p, r, f1))

    arr = np.array(rows, dtype=object)
    metrics = arr[:, 2:].astype(float)

    print("\n[CV|Baseline] === Summary über alle Folds ===")
    print(f"[CV|Baseline] mean -> acc={metrics[:,0].mean():.4f} p={metrics[:,1].mean():.4f} "
          f"r={metrics[:,2].mean():.4f} f1={metrics[:,3].mean():.4f}")
    print(f"[CV|Baseline]  std -> acc={metrics[:,0].std(ddof=1):.4f} p={metrics[:,1].std(ddof=1):.4f} "
          f"r={metrics[:,2].std(ddof=1):.4f} f1={metrics[:,3].std(ddof=1):.4f}")

    cv_df = pd.DataFrame(rows, columns=["run","fold","accuracy","precision_macro","recall_macro","f1_macro"])
    cv_df.loc[len(cv_df)] = ["mean","",metrics[:,0].mean(),metrics[:,1].mean(),metrics[:,2].mean(),metrics[:,3].mean()]
    cv_df.loc[len(cv_df)] = ["std","",metrics[:,0].std(ddof=1),metrics[:,1].std(ddof=1),metrics[:,2].std(ddof=1),metrics[:,3].std(ddof=1)]
    cv_df.to_csv("cv_baseline_summary.csv", index=False, encoding="utf-8")
    print("[CV|Baseline] Gespeichert: cv_baseline_summary.csv")
    return cv_df


# Cross-Validation: BERT (distilbert-base-uncased)
from datasets import Dataset
from transformers import DataCollatorWithPadding


def run_bert_cv(model_name="distilbert-base-uncased", max_len=512, k=3, repeats=1, seed=42, data_dir="data"):
    print("\n[CV|BERT] Lade train.csv für Cross-Validation …")
    train_df = _read_csv(os.path.join(data_dir, "train.csv")).rename(columns={"y":"label"})
    X = train_df["text"].to_numpy()
    y = train_df["label"].to_numpy().astype(int)
    print(f"[CV|BERT] Train-Set: {len(X)} Beispiele | Klassenverteilung: "
          f"{(y==0).sum()} non-toxic, {(y==1).sum()} toxic")
    print(f"[CV|BERT] Modell: {model_name} | max_len={max_len} | folds={k}")

    tok = AutoTokenizer.from_pretrained(model_name)
    fp16 = torch.cuda.is_available()
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8 if fp16 else None)

    # Nur Header-Strip (keine Lemmas) für BERT
    def tokenize(batch):
        cleaned = [strip_subreddit_header(t) for t in batch["text"]]
        return tok(cleaned, truncation=True, max_length=max_len)

    # Repeated Stratified K-Fold optional
    if repeats and repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
        total_folds = k * repeats
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        total_folds = k

    metrics = []

    for global_fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        run_idx = 1 if repeats <= 1 else int(np.ceil(global_fold / k))
        fold_in_run = ((global_fold - 1) % k) + 1

        print(f"\n[CV|BERT] === Run {run_idx}/{repeats if repeats>1 else 1} | Fold {fold_in_run}/{k} "
              f"(global {global_fold}/{total_folds}) ===")
        print(f"[CV|BERT] Train-Indices: {len(tr_idx)} | Val-Indices: {len(va_idx)}")

        df_tr = pd.DataFrame({"text": X[tr_idx], "label": y[tr_idx]})
        df_va = pd.DataFrame({"text": X[va_idx], "label": y[va_idx]})

        ds_tr = Dataset.from_pandas(df_tr, preserve_index=False).map(tokenize, batched=True)
        ds_va = Dataset.from_pandas(df_va, preserve_index=False).map(tokenize, batched=True)

        keep = ["input_ids","attention_mask","label"]
        ds_tr = ds_tr.remove_columns([c for c in ds_tr.column_names if c not in keep]).with_format("torch")
        ds_va = ds_va.remove_columns([c for c in ds_va.column_names if c not in keep]).with_format("torch")

        cw = torch.tensor(
            compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=df_tr["label"].to_numpy()),
            dtype=torch.float
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="single_label_classification")

        def compute_metrics_cv(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}

        class WTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                loss_fct = torch.nn.CrossEntropyLoss(weight=cw.to(outputs.logits.device))
                loss = loss_fct(outputs.logits, labels)
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=f"./out_cv/bert_fold{fold_in_run}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="no",
            seed=seed,
            fp16=fp16,
            report_to="none",
            logging_steps=50,
        )

        print("[CV|BERT] Starte Training …")
        trainer = WTrainer(
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics_cv,
        )
        trainer.train()

        print("[CV|BERT] Eval auf Fold-Validierung …")
        res = trainer.evaluate(eval_dataset=ds_va)
        print(f"[CV|BERT] Fold {fold_in_run} -> acc={res['eval_accuracy']:.4f} "
              f"p={res['eval_precision_macro']:.4f} r={res['eval_recall_macro']:.4f} f1={res['eval_f1_macro']:.4f}")
        metrics.append([res['eval_accuracy'], res['eval_precision_macro'], res['eval_recall_macro'], res['eval_f1_macro']])

    m = np.array(metrics, dtype=float)
    print("\n[CV|BERT] === Summary über alle Folds ===")
    print(f"[CV|BERT] mean -> acc={m[:,0].mean():.4f} p={m[:,1].mean():.4f} r={m[:,2].mean():.4f} f1={m[:,3].mean():.4f}")
    print(f"[CV|BERT]  std -> acc={m[:,0].std(ddof=1):.4f} p={m[:,1].std(ddof=1):.4f} r={m[:,2].std(ddof=1):.4f} f1={m[:,3].std(ddof=1):.4f}")

    cv_df = pd.DataFrame(metrics, columns=["accuracy","precision_macro","recall_macro","f1_macro"])
    cv_df.insert(0, "fold", np.arange(1, len(cv_df)+1))
    cv_df.loc[len(cv_df)] = ["mean", m[:,0].mean(), m[:,1].mean(), m[:,2].mean(), m[:,3].mean()]
    cv_df.loc[len(cv_df)] = ["std",  m[:,0].std(ddof=1), m[:,1].std(ddof=1), m[:,2].std(ddof=1), m[:,3].std(ddof=1)]
    cv_df.to_csv("cv_bert_summary.csv", index=False, encoding="utf-8")
    print("[CV|BERT] Gespeichert: cv_bert_summary.csv")
    return cv_df


# Cross-Validation
print("\n>>> STARTE CROSS-VALIDATION (Baseline) …")
_ = run_baseline_cv(data_dir="data", k=5, repeats=3, seed=SEED)

print("\n>>> STARTE CROSS-VALIDATION (BERT) …")
_ = run_bert_cv(model_name=MODEL_NAME, max_len=256, k=3, repeats=3, seed=SEED, data_dir="data")

print("\n>>> CROSS-VALIDATION abgeschlossen.")
