# IL2
Dieses Repository enthält Code zur Klassifikation von toxischen Texten in Reddit-Beiträgen. Es kombiniert eine klassische Baseline (TF-IDF + Logistic Regression) mit Transformer-basierten Modellen (DistilBERT).
Enthalten ist ein Tool zur manuellen Annotation, Preprocessing-Funktionen, sowie Training und Evaluation mit Cross-Validation. Ergebnisse werden in CSV-Dateien gespeichert. Das Projekt dient Forschungszwecken und ersetzt keine professionelle Beratung oder Diagnose.

Alle Libraries:
pandas – Datenhandling (CSV einlesen, Annotationen speichern)
numpy – Arrays, Zufallszahlen (Seed setzen, Data Handling)
scikit-learn – Baseline-Modelle & Preprocessing
TfidfVectorizer, LogisticRegression, classification_report, confusion_matrix, StratifiedKFold
datasets (Hugging Face) – Reddit-Daten laden (load_dataset("trl-lib/tldr"))
transformers (Hugging Face) – DistilBERT Modell, Tokenizer, Trainer
torch (PyTorch) – Framework für Transformers (Training, GPU Nutzung)
spacy – Tokenization/Preprocessing (en_core_web_sm Modell nötig)
random – Seed setzen
os – Dateioperationen (annotations.csv speichern)
