# address_parser.py
# Lightweight address parser with no external databases or libpostal dependency.
import joblib

_model = joblib.load("address_parser.joblib")

def parse(address: str):
    return _model.parse(address)

def add_examples(X, y):
    return _model.add_examples(X, y)

def replace_examples(X, y):
    return _model.replace_examples(X, y)

def get_metrics():
    return _model.training_metrics
