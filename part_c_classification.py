import numpy as np
import pandas as pd

from src.shipment_prep import print_stats
from src.shipment_prep import FEATURES
from src.shipment_prep import NOISE_COLUMNS

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
def get_metrics(model, actual, predicted):
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    #############################################################################
    # The four confusion-matrix cells in each row must reproduce accuracy, precision, recall and f1
    #############################################################################
    cal_accuracy = round((tn + tp) / (tn + tp + fp + fn), 4)
    cal_precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    cal_recall = round(tp / (tp + fn), 4)
    cal_f1 = round((2 * cal_precision * cal_recall) / (cal_precision + cal_recall), 4) if (cal_precision + cal_recall) > 0 else 0.0
    #############################################################################
    metrics = {
            "model": model,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": round(accuracy_score(actual, predicted), 4),
            #"cal_accuracy": cal_accuracy, 
            "precision": round(precision_score(actual, predicted, zero_division=0), 4),
            #"cal_precision": cal_precision, 
            "recall": round(recall_score(actual, predicted), 4),
            #"cal_recall": cal_recall, 
            "f1": round(f1_score(actual, predicted), 4), 
            #"cal_f1": cal_f1, 
        }
    metrics_df = pd.DataFrame([metrics])
    return metrics_df

#######################
#Create Baseline
#blind_all_on_time (predicts "not delayed" for every parcel; write precision as 0)
conslidated_df = pd.DataFrame()

y_baseline_pred = np.full_like(y_val, fill_value=0)
baseline_metrics_df = get_metrics("blind_all_on_time", y_val, y_baseline_pred)
#baseline mean
conslidated_df = pd.concat([conslidated_df, baseline_metrics_df], ignore_index=True)

#LogisticRegression
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()), 
    ("model", LogisticRegression())
    ])
pipeline_lr.fit(X_train, y_train)
y_pred_val = pipeline_lr.predict(X_val)
result_df = get_metrics("logistic_regression", y_val, y_pred_val)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)

###########################################
#Tree-depth diagnostic.
def tree_depth_diagnostic(d, X_train, y_train, X_val, y_val):    
    model = DecisionTreeClassifier(max_depth=d, criterion="gini", random_state=42)
    model.fit(X_train, y_train)    
    y_pred_val = model.predict(X_val)
    df = get_metrics("decision_tree_depth_" + str(d), y_val, y_pred_val)
    return df

depths = [2, 3, 4, 5, 6, 8, 10]
highest_f1_df = pd.DataFrame()
for d in depths:
    result_df = tree_depth_diagnostic(d, X_train, y_train, X_val, y_val)
    #print(result_df)
    #with d chosen from 2, 3, 4, 5, 6, 8, 10 by highest validation F1
    if(len(highest_f1_df) == 0):
        highest_f1_df = result_df
    elif(result_df["f1"].iloc[0] > highest_f1_df["f1"].iloc[0]):
        highest_f1_df = result_df        
    #print(highest_f1_df)
    #conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)

conslidated_df = pd.concat([conslidated_df, highest_f1_df], ignore_index=True)
#RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
result_df = get_metrics("random_forest_default", y_val, y_pred_val)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)

#print(conslidated_df)
#write to csv file
conslidated_df.to_csv("outputs/part_c_models.csv", index=False)
print("outputs/part_c_models.csv generated successfully...")

#Task 3: Which columns carry signal? Fit a default forest (n_estimators=100, random_state=42) on 
# all 17 columns of the training rows to get feature_importances_; fit SelectKBest(f_classif, k="all") on 
# the training rows to get one F-score per column. Write outputs/part_c_feature_scores.csv 
# (feature, f_score, f_rank, rf_importance, is_noise; f_rank 1 = highest F-score; is_noise is True for 
# the five NOISE_COLUMNS). Then, for k in 5, 7, 10, 12, 17, fit a Pipeline of SelectKBest(f_classif, k=k) 
# followed by RandomForestClassifier(n_estimators=100, random_state=42) on the training rows and 
# write the validation F1 to outputs/part_c_selectk.csv (k, val_f1). From here on k = 7.
#print(X_train.shape)

#RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
rf_importances = model.feature_importances_
#print("rf_importances: ", rf_importances)

# Fit SelectKBest to get ANOVA F-scores
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)
f_scores = selector.scores_
#print("f_scores: ", f_scores)

#combine into DataFrame
output_df = pd.DataFrame({
    "feature": X_train.columns,
    "f_score": f_scores.round(4),
    "rf_importance": rf_importances.round(4)
})

# Add the f_rank column (highest f_score gets rank 1)
output_df["f_rank"] = output_df["f_score"].rank(ascending=False, method="min").astype(int)
output_df = output_df.sort_values(by="f_rank")

# Create the is_noise column: True if the feature is in NOISE_COLUMNS, False otherwise
output_df["is_noise"] = output_df["feature"].isin(NOISE_COLUMNS)
#rearrange the columns
new_order = ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
output_df = output_df[new_order]
#print(output_df)
#write to csv file
output_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
print("outputs/part_c_feature_scores.csv generated successfully...")

#Then, for k in 5, 7, 10, 12, 17, fit a Pipeline of SelectKBest(f_classif, k=k) 
# followed by RandomForestClassifier(n_estimators=100, random_state=42) on the training rows and 
# write the validation F1 to outputs/part_c_selectk.csv (k, val_f1).
#RandomForestClassifier
k_val = [5, 7, 10, 12, 17]
conslidated_df = pd.DataFrame()
for k in k_val:
    pipeline_rf = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=k)), 
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    pipeline_rf.fit(X_train, y_train)
    y_pred_val = pipeline_rf.predict(X_val)
    val_f1 = round(f1_score(y_val, y_pred_val), 4)
    val_df = pd.DataFrame([{"k":k, "val_f1": val_f1}])
    conslidated_df = pd.concat([conslidated_df, val_df], ignore_index=True)

#print(conslidated_df)
#write to csv file
conslidated_df.to_csv("outputs/part_c_selectk.csv", index=False)
print("outputs/part_c_selectk.csv generated successfully...")

#Task 4: Tuning. Fit SelectKBest(f_classif, k=7) on the training rows only and transform the training 
# and validation rows with it. On the transformed training rows run, for RandomForestClassifier(random_state=42),
# over the grid n_estimators ∈ {50, 100, 150}, max_depth ∈ {4, 6, 8}, min_samples_leaf ∈ {1, 5, 10}: 
# GridSearchCV(..., cv=5) and RandomizedSearchCV(..., n_iter=9, cv=5, random_state=42) (default scoring for both). 
# Write outputs/part_c_tuning.csv with columns search, candidates, cv_folds, total_fits, best_params, 
# best_cv_accuracy, val_f1 and three rows: grid_search, randomized_search, default_forest_same_7_features 
# (the untuned forest n_estimators=100, no depth limit, min_samples_leaf=1 on the same 7 columns; candidates 1, 
# cv_folds 0, total_fits 1, best_params = defaults, best_cv_accuracy left blank). 
# total_fits = candidates × cv_folds for the two searches; val_f1 is each candidate's F1 on the validation rows. 
# The test rows must not be touched in this task.

#Fit SelectKBest(f_classif, k=7) on the training rows only and transform the training 
# and validation rows with it.
selector = SelectKBest(score_func=f_classif, k=7)
selector.fit(X_train, y_train)

X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.fit_transform(X_val, y_val)

#On the transformed training rows run, for RandomForestClassifier(random_state=42),
# over the grid n_estimators ∈ {50, 100, 150}, max_depth ∈ {4, 6, 8}, min_samples_leaf ∈ {1, 5, 10}: 
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}
#####################################
#GridSearchCV(..., cv=5)
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5
)
grid_search.fit(X_train_selected, y_train)
val_f1 = f1_score(y_val, grid_search.best_estimator_.predict(X_val_selected))
gs_df = pd.DataFrame([{
    "columns": X_train_selected.shape[1],
    "search": "grid_search",
    "candidates": 3 * 3 * 3,
    "cv_folds": 5,
    "total_fits": 3 * 3 * 3 * 5,
    "best_params": grid_search.best_params_,
    "best_cv_accuracy": round(grid_search.best_score_, 4),
    "val_f1": round(val_f1, 4)
    }])
###############################################
#RandomizedSearchCV(..., n_iter=9, cv=5, random_state=42)
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_grid,
    n_iter=9, cv=5, random_state=42
)
random_search.fit(X_train_selected, y_train)
val_f1 = f1_score(y_val, random_search.best_estimator_.predict(X_val_selected))
rs_df = pd.DataFrame([{
    "columns": X_train_selected.shape[1],
    "search": "randomized_search",
    "candidates": 9,
    "cv_folds": 5,
    "total_fits": 9 * 5,
    "best_params": random_search.best_params_,
    "best_cv_accuracy": round(random_search.best_score_, 4),
    "val_f1": round(val_f1, 4)
    }])

#default_forest_same_7_features 
# (the untuned forest n_estimators=100, no depth limit, min_samples_leaf=1 on the same 7 columns; candidates 1, 
# cv_folds 0, total_fits 1, best_params = defaults, best_cv_accuracy left blank). 
model = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_leaf=1, random_state=42)
model.fit(X_train_selected, y_train)

y_pred_val = model.predict(X_val_selected)
acc_score = accuracy_score(y_val, y_pred_val)
val_f1 = f1_score(y_val, y_pred_val)
default_df = pd.DataFrame([{
    "columns": X_train_selected.shape[1],
    "search": "default_forest_same_7_features",
    "candidates": 1,
    "cv_folds": 0,
    "total_fits": 1,
    "best_params": "defaults",
    "best_cv_accuracy": round(acc_score, 4),
    "val_f1": round(val_f1, 4)
    }])

conslidated_df = pd.DataFrame()
conslidated_df = pd.concat([conslidated_df, default_df], ignore_index=True)
conslidated_df = pd.concat([conslidated_df, gs_df], ignore_index=True)
conslidated_df = pd.concat([conslidated_df, rs_df], ignore_index=True)

#print(gs_df)
#write to csv file
conslidated_df.to_csv("outputs/part_c_tuning.csv", index=False)
print("outputs/part_c_tuning.csv generated successfully...")

######################################################################
#Task 5: Choosing the final configuration. Pick the row of part_c_tuning.csv with the highest val_f1 
# (ties → default_forest_same_7_features, then grid_search, then randomized_search). The final model 
# is a Pipeline([("selector", SelectKBest(f_classif, k=7)), 
# ("model", RandomForestClassifier(random_state=42, <chosen parameters>))]) fitted on the training rows.

pipeline_rf_final = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=7)), 
    ("model", RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=6, min_samples_leaf=1, random_state=42))
    ])
pipeline_rf_final.fit(X_train, y_train)
y_prob_val = pipeline_rf_final.predict_proba(X_val)
#print(y_prob_val)

############################################################
#Task 6: Alert threshold. Take the final model's validation probabilities (predict_proba, positive class). 
# Build the thresholds as exact decimals — use the literal list [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
# or round each value to one decimal (a floating-point range such as np.arange(0.1, 1.0, 0.1) produces values 
# like 0.30000000000000004 that change the result at the boundary). For each threshold t, predict "delayed" 
# when the probability is ≥ t, and write outputs/part_c_thresholds.csv with columns threshold, TN, FP, FN, TP, 
# accuracy, precision, recall, f1. Operations rule: because a missed delayed parcel costs more than an 
# unnecessary call, the alert threshold is the highest t whose validation recall is at least 0.80 
# (if none reaches 0.80, use 0.1).

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
metrics_list = []
#print(y_val)
#print("y_val.shape: ", y_val.shape)
# calculate performance metrics for each threshold configuration
for t in thresholds:
    # Predict "delayed" (1) if probability is >= t, else 0
    y_pred = (y_prob_val[:, 1] >= t).astype(int)
    #print(y_pred)
    #print("y_pred.shape: ", y_pred.shape)
    # Extract confusion matrix tracking
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    # metrics
    acc = round(accuracy_score(y_val, y_pred), 4)
    prec = round(precision_score(y_val, y_pred, zero_division=0), 4)
    rec = round(recall_score(y_val, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_val, y_pred, zero_division=0), 4)
    # add details
    metrics_list.append({
        "threshold": t,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

df_thresholds = pd.DataFrame(metrics_list)

#write to csv file
df_thresholds.to_csv("outputs/part_c_thresholds.csv", index=False)
print("outputs/part_c_thresholds.csv generated successfully...")

####################################################################3
#Task 7: One test evaluation. Score the final model on the test rows at the chosen threshold, 
# exactly once, and write outputs/part_c_final_test.csv (threshold, TN, FP, FN, TP, accuracy, precision, 
# recall, f1). Write outputs/part_c_final_config.csv, one row, columns selector_k, n_estimators, max_depth, 
# min_samples_leaf, threshold, chosen_from (max_depth is unlimited when there is no limit; 
# chosen_from is the name of the winning candidate in part_c_tuning.csv).

pipeline_rf_final = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=7)), 
    ("model", RandomForestClassifier(n_estimators=50, criterion="gini", max_depth=6, min_samples_leaf=1, random_state=42))
    ])
pipeline_rf_final.fit(X_train, y_train)
y_prob_test = pipeline_rf_final.predict_proba(X_test)
##################
#highest t whose validation recall is at least 0.80
# t = 0.4 (Recall = 0.8571)
##################
choosen_t = 0.4
##################
y_pred_test = (y_prob_test[:, 1] >= choosen_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
# metrics
acc = round(accuracy_score(y_test, y_pred_test), 4)
prec = round(precision_score(y_test, y_pred_test, zero_division=0), 4)
rec = round(recall_score(y_test, y_pred_test, zero_division=0), 4)
f1 = round(f1_score(y_test, y_pred_test, zero_division=0), 4)
# add details
final_list = []
final_list.append({
    "threshold": choosen_t,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1
})

df_final_test = pd.DataFrame(final_list)
#write to csv file
df_final_test.to_csv("outputs/part_c_final_test.csv", index=False)
print("outputs/part_c_final_test.csv generated successfully...")

#Write outputs/part_c_final_config.csv, one row, columns selector_k, n_estimators, max_depth, 
# min_samples_leaf, threshold, chosen_from (max_depth is unlimited when there is no limit; 
# chosen_from is the name of the winning candidate in part_c_tuning.csv).
final_config = []
final_config.append({
    "columns": X_test.shape[1],
    "selector_k": 7,
    "n_estimators": 50,
    "max_depth": 6,
    "min_samples_leaf": 1,
    "threshold": 0.4,
    "chosen_from": "randomized_search"
})
df_final_config = pd.DataFrame(final_config)
#write to csv file
df_final_config.to_csv("outputs/part_c_final_config.csv", index=False)
print("outputs/part_c_final_config.csv generated successfully...")
