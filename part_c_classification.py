import numpy as np
import pandas as pd

from src.shipment_prep import print_stats
from src.shipment_prep import FEATURES

from sklearn.model_selection import train_test_split

#Tasks — all in part_c_classification.py, reading data/modelling_table.csv. 
# Inputs are the 17 FEATURES in the pinned order; the target is is_delayed.

#load csv file and build dataframe
data_df = pd.read_csv("data/modelling_table.csv")
#print_stats(data_df)

X = data_df[FEATURES]
y = data_df["is_delayed"]

#############################################################################
#Task 1: Stratified split. Repeat Part B's two-step split (test_size=0.3 then test_size=0.5, random_state=42 both times) 
# with stratify on the target in both calls. Expected sizes 840 / 180 / 180 with 195 / 42 / 41 delayed parcels.
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print(f"X_train: {len(X_train)} X_val: {len(X_val)}, X_test: {len(X_test)}")
print(f"y_train: {(y_train == 1).sum()} y_val: {(y_val == 1).sum()}, y_test: {(y_test == 1).sum()}")

#############################################################################
#Task 2: Blind baseline and three classifiers, scored on the validation rows with the 
# default 0.5 decision rule. Write outputs/part_c_models.csv with columns model, TN, FP, FN, TP, 
# accuracy, precision, recall, f1 and exactly these rows: blind_all_on_time (predicts "not delayed" 
# for every parcel; write precision as 0); logistic_regression (a Pipeline of StandardScaler 
# then LogisticRegression); decision_tree_depth_<d> (DecisionTreeClassifier(criterion="gini", 
# max_depth=d, random_state=42), with d chosen from 2, 3, 4, 5, 6, 8, 10 by highest validation F1, 
# ties → smaller d, and <d> replaced by the chosen value); random_forest_default 
# (RandomForestClassifier(n_estimators=100, random_state=42)). 
# The four confusion-matrix cells in each row must reproduce accuracy, precision, recall and f1