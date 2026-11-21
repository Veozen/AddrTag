# address parsing using conditional random fields(CRF)
import re
import unicodedata
import math
from collections import defaultdict
import numpy as np
import joblib


import sklearn_crfsuite
from sklearn.model_selection import GridSearchCV, KFold ,cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn_crfsuite import metrics
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression


POSTAL_CODE_REGEX = re.compile(r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$')
FSA_REGEX = re.compile(r"^[A-Za-z][0-9][A-Za-z]$")
LDU_REGEX = re.compile(r"^[0-9][A-Za-z][0-9]$")

POSTAL_REGEXES = [
    r"^\d{5}$",                     # USA, France, Germany (5 digits)
    r"^\d{5}-\d{4}$",               # USA ZIP+4
    r"^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$", # Canada
    r"^[A-Za-z][0-9][A-Za-z]$",             # Canada FSA
    r"^[0-9][A-Za-z][0-9]$"   ,             # Canada LDU

    r"^\d{4}\s?[A-Za-z]{2}$",       # Netherlands (1234 AB)
    r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$",  # UK
    r"^\d{3}-\d{4}$",               # Japan (123-4567)
    r"^\d{4}$",                     # Australia, Denmark, Austria
    r"^\d{6}$",                     # Singapore, Russia, India (some)
    r"^\d{3}\s?\d{2}$",             # Sweden (123 45)
    r"^\d{2}-\d{3}$",               # Poland (12-345)
]

POSTAL_PATTERNS = [re.compile(p) for p in POSTAL_REGEXES]


# Feature extraction and preprocessing
def remove_accents(text):
    # remove (){}[]
    for ch in "(){}[]":
        text = text.replace(ch, " ")
    # Normalize to "NFD" form: separates base letters from accents
    normalized = unicodedata.normalize("NFD", text)
    # Filter out combining marks (accents)
    return "".join(c for c in normalized if not unicodedata.combining(c))

def normalize_token(t):
    t = t.lower()
    t = t.replace(".", "")  #remove .
    return remove_accents(t)

def replace_nonword_dashes(text):
    # Split # from unit number
    text = re.sub(r'#(?=\w)', '# ', text)

    # 1. digit - digit
    text = re.sub(r'(?<=\d)-(?=\d)', ' ', text)

    # 2. digit - letter
    text = re.sub(r'(?<=\d)-(?=[A-Za-z])', ' ', text)

    # 3. letter - digit
    text = re.sub(r'(?<=[A-Za-z])-(?=\d)', ' ', text)

    return text

def split_address(address: str):
    """
    Splits an address only on dashes that appear between two digits.
    Leaves word-dashes (e.g., 'North-West') untouched.
    """
    # Replace digit-digit dashes with a space
    cleaned = replace_nonword_dashes(address)

    # remove ,
    cleaned = cleaned.replace(",", " ") #remove ,

    # Split on whitespace
    return cleaned.split()

def build_lexicon(training_data, target_label):
    """
    training_data:
    target_label: the label to extract tokens for (e.g., "PROVINCE")
    """
    lexicon = set()
    for tokens, labels in training_data:
        for tok, lab in zip(tokens, labels):
            if lab == target_label:
                lexicon.add(normalize_token(tok))
    return lexicon

def matches_any_postal_regex(token):
    for pattern in POSTAL_PATTERNS:
        if pattern.match(token):
            return True
    return False

def extract_features(tokens, lexicons=None):
    """Extract expanded features for each token."""
    features = []

    for i, token in enumerate(tokens):
        token_features = {
            'bias': 1.0,  # Bias term
            #'token': normalize_token(token),
            "has_digit" : any(ch.isdigit() for ch in token),
            "has_alpha" : any(ch.isalpha() for ch in token),
            "is_postal_candidate": matches_any_postal_regex(token),
            'is_first': i==0,
            'is_last': i==len(tokens),
        }
        # Add lexicon-based features
        for lexicon_name, lexicon_set in lexicons.items():
            token_features[f"in_{lexicon_name}_lexicon"] = normalize_token(token) in lexicon_set

        features.append(token_features)
    return features

def group_labels(address_tokens,labels):
    # Group consecutive tokens with the same label
    grouped         = {}
    current_label   = None
    current_tokens  = []

    for token, label in zip(address_tokens, labels):
        if label != current_label:
            # flush previous group
            if current_label is not None:
                grouped.setdefault(current_label, []).append(" ".join(current_tokens))
            current_label = label
            current_tokens = [token]
        else:
            current_tokens.append(token)

    # flush last group
    if current_label is not None:
        grouped.setdefault(current_label, []).append(" ".join(current_tokens))

    return grouped #dict[label: grouped_token]

def validate_examples(X,y):
    # Same number of training examples
    if len(X) != len(y):
        raise ValueError(
            f"X contains {len(X)} examples but y contains {len(y)} label sequences."
        )

    # Validate token/label alignment
    for i, (sentence, labels) in enumerate(zip(X, y)):

        if len(sentence) != len(labels):
            raise ValueError(
                f"Example {i}: found {len(sentence)} tokens but "
                f"{len(labels)} labels.\n"
                f"Sentence: {sentence}\n"
                f"Labels: {labels}"
            )

#calibration
def cross_val_sequence_probs(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    seq_probs = []
    correctness = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

        # Fit on training fold
        model.fit(X_train, y_train)

        # Predict on test fold
        y_pred = model.predict(X_test)
        tagger = model.tagger_

        for features, true_labels, pred_labels in zip(X_test, y_test, y_pred):
            # Compute CRF sequence probability
            tagger.set(features)
            prob = tagger.probability(pred_labels)

            seq_probs.append(prob)
            correctness.append(int(pred_labels == true_labels))

    return np.array(seq_probs), np.array(correctness)

def linear_interp(x,x_vals,y_vals):
    return np.interp(x, x_vals, y_vals)


class AddrParse(BaseEstimator):
    """
    nearest neighboor address parser
    predictor features have to be binary either True/False or 0/1
    """
    def __init__(self, lexicon=None,
                        c1=0,
                        c2=0):
        self.labels = None
        self.lexicon = lexicon
        self.c1 = c1
        self.c2 = c2

        self.crf_parser = sklearn_crfsuite.CRF( algorithm='lbfgs',
                                                max_iterations=100,
                                                all_possible_transitions=True,
                                                c1 = self.c1,
                                                c2 = self.c2
                                            )

    def fit(self, X, y, verbose = False):
        """
        X: list of sentences
        y: list of label sequences
        """

        validate_examples(X,y)

        print("Number of examples used: ", len(y))

        #store tarining data
        self.X_train = X
        self.y_train = y

        #collect all labels
        self.labels = set()
        for labels in y:
            for lab in labels:
                self.labels.add(lab)

        if not self.lexicon:
            self.lexicon = labels.copy()

        # Recombine into the structure your build_lexicon() expects
        train_pairs = list(zip(X, y))

        # X is a list of sentences (or sentence/label pairs depending on your structure)
        self.lexicons = { label: build_lexicon(train_pairs, label) for label in self.lexicon if label in self.labels}

        X_feat = [extract_features(sentence, self.lexicons) for sentence in X]

        # 2. Obtain out-of-fold predictions
        seq_probs, correctness = cross_val_sequence_probs( self.crf_parser,
                                                            X_feat,
                                                            y,
                                                            n_splits=5)

        # 3. Fit calibrator
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(seq_probs, correctness)


        # record model performance
        y_pred = cross_val_predict(
            self.crf_parser,
            X_feat,
            y,
            cv=5
        )

        # flatten sequences
        y_true_flat = [lab for seq in y for lab in seq]
        y_pred_flat = [lab for seq in y_pred for lab in seq]

        self.training_metrics = {
            "accuracy": accuracy_score(y_true_flat, y_pred_flat),
            "f1_macro": f1_score(
                y_true_flat,
                y_pred_flat,
                average="macro"
            )
        }

        if verbose:
            print(
                f"Accuracy={self.training_metrics['accuracy']:.4f} "
                f"F1={self.training_metrics['f1_macro']:.4f}"
            )

        #fit model
        self.crf_parser.fit(X_feat, y)


    def add_examples(self, X_new, y_new):

        validate_examples(X_new, y_new)

        print("Number of examples added: ", len(y_new))

        self._X_train.extend(X_new)
        self._y_train.extend(y_new)

        self.fit(self._X_train, self._y_train)


    def replace_examples(self, X_new, y_new):

        validate_examples(X_new, y_new)

        print("Number of examples added: ", len(y_new))

        self._X_train = X_new
        self._y_train = y_new

        self.fit(self._X_train, self._y_train)

    def predict(self, X):
        addresses_feat = [extract_features(sentence, self.lexicons) for sentence in X]

        return self.crf_parser.predict(addresses_feat)

    def score(self, X, y):
        pred = self.predict(X)
        # print("len(y):", len(y))
        # print("len(pred):", len(pred))

        return metrics.flat_accuracy_score(y, pred)

    def parse(self, addr_str):
        address_tokens = split_address(addr_str)
        address_features = extract_features(address_tokens, self.lexicons)

        labels = self.crf_parser.predict([address_features])[0]

        marginals = self.crf_parser.predict_marginals_single(address_features)

        tagger = self.crf_parser.tagger_
        tagger.set(address_features)
        score = tagger.probability(labels)

        #calibrated score using linear interpolation
        prob = float(linear_interp( [score],
                                    self.calibrator.X_thresholds_,
                                    self.calibrator.y_thresholds_)[0])
        #prob = float(self.calibrator([score])[0])

        grouped_tokens = group_labels(address_tokens,labels)

        solution = {"parsed": grouped_tokens,
                    "score": score,
                    "confidence": prob,
                    "tokens": address_tokens }

        return solution


# Prepare training data (tuples)
training_data = [ y for x,y in addresses ]


# Create feature and label datasets
X_inputs = [ [tup[0] for tup in tuple_lst]
            for tuple_lst in training_data]
y_labels = [ [tup[1] for tup in tuple_lst]
            for tuple_lst in training_data]



# Prepare training data
training_data = [ (split_address(x),y)
                    for x,y in addresses
                    if len(split_address(x)) == len(y)]


# Create feature and label datasets
X_inputs = [ sentence
            for sentence, labels in training_data]
y_labels = [ labels
            for sentence, labels in training_data]



#find best hyperparams
param_grid = {
    'c1': [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56],
    'c2': [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56],
    'lexicon': [["Province", "StreetType", "StreetDirection", "UnitType"],
                ["Province", "StreetType", "StreetDirection", "UnitType","Quadrant", "QuarterSection","RuralRoute","RecipientMarker","POBox","Country","State"],
                ]
}


search = GridSearchCV( AddrParse(),
                        param_grid,
                        cv=3)

# Create feature and label datasets
search.fit(X_inputs, y_labels)

best_model = search.best_estimator_
best_params = search.best_params_
best_score_ = search.best_score_
best_model.score(X_inputs, y_labels)

# Save
joblib.dump(best_model, "address_parser/data/addr_parser.joblib")

# Save
joblib.dump(best_model, "addr_parser.joblib")
