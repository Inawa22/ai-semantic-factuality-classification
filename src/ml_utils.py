import re
import json
import numpy as np
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())

def get_custom_stopwords():
    return set(ENGLISH_STOP_WORDS).difference({"not", "no", "never"})

def get_tfidf_config():
    return {
        "stop_words": list(get_custom_stopwords()),
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
        "max_features": 200_000,
    }

def save_tfidf_config(path="tfidf_config.json"):
    with open(path, "w") as f:
        json.dump(get_tfidf_config(), f, indent=2)

def save_similarities(qa_sim, ca_sim, path="similarity_features.npz"):
    np.savez(path, qa_sim=qa_sim, ca_sim=ca_sim)

def load_similarities(path="similarity_features.npz"):
    data = np.load(path)
    return data["qa_sim"], data["ca_sim"]

def save_model(model, path="final_model.joblib"):
    joblib.dump(model, path)

def load_model(path="final_model.joblib"):
    return joblib.load(path)
EOF
