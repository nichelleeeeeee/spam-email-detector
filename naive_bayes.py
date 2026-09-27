import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix


DATA_FILE = "data.csv"

#Load the data
df = pd.read_csv(DATA_FILE, encoding = 'latin-1',usecols=[0, 1])
df.columns = ["label", "text"]

df["label"] = df["label"].astype(str).str.lower().str.strip()

#Separate emails and labels
X = df["text"].values
y = df["label"].values

skf = StratifiedKFold(
  n_splits=5,
  shuffle=True,
  random_state=1
)

nb = make_pipeline(
  TfidfVectorizer(),
  MultinomialNB()
)

lst_accu_stratified = []
lst_f1_stratified = []
lst_fp_stratified = []
lst_fn_stratified = []

for train_index, test_index in skf.split(X, y):

    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    nb.fit(X_train_fold, y_train_fold)

    lst_accu_stratified.append(nb.score(X_test_fold, y_test_fold))
    lst_f1_stratified.append(f1_score(y_test_fold,nb.predict(X_test_fold), pos_label="spam"))
    
    tn, fp, fn, tp = confusion_matrix( y_test_fold,nb.predict(X_test_fold),labels=["ham", "spam"]).ravel()
    lst_fn_stratified.append(fn)
    lst_fp_stratified.append(fp)

#Output
print("Naive Bayes")
for i in range(5):
  print(f"fold {i+1} accuracy: {lst_accu_stratified[i]:.2%}")
  print(f"fold {i+1} F1 score: {lst_f1_stratified[i]:.2%}")
  print(f"fold {i+1} false positives: {lst_fp_stratified[i]}")
  print(f"fold {i+1} False negatives: {lst_fn_stratified[i]}")

