import numpy as np
import pandas as pd

from src.shipment_prep import print_stats
from src.shipment_prep import FEATURES

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load csv file and build dataframe
data_df = pd.read_csv("data/modelling_table.csv")
#print_stats(data_df)

######################################################
#Task 1: Split into training, validation and test. Call train_test_split with test_size=0.3, random_state=42 to 
# separate 70% for training from a temporary 30%; call it again on the temporary part with test_size=0.5, 
# random_state=42 to get validation and test. Sizes must be 840 / 180 / 180.
X = data_df[FEATURES]
y = data_df["delivery_hours"]

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
print(f"Train: {len(X_train)} Validation: {len(X_val)}, Test: {len(X_test)}")

######################################################
#Task 2: Scaling without leakage. Fit a StandardScaler on the 840 training rows only, 
# then transform the test rows with that fitted scaler. Write outputs/part_b_checks.csv (metric,value) 
# with train_rows, val_rows, test_rows, train_scaled_max_abs_mean (largest absolute column mean of the 
# scaled training rows) and test_scaled_max_abs_mean (the same for the scaled test rows).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

part_b_checks_file = {}
part_b_checks_file["train_rows"] = len(X_train)
part_b_checks_file["val_rows"] = len(X_val)
part_b_checks_file["test_rows"] = len(X_test)
#print(X_train_scaled)
abs_col_means = np.abs(X_train_scaled).mean(axis=0)
#print(abs_col_means)
max_idx = np.argmax(abs_col_means)
largest_col_name = X_train.columns[max_idx]
largest_val = abs_col_means[max_idx]
part_b_checks_file["train_scaled_max_abs_mean_column"] = largest_col_name
part_b_checks_file["train_scaled_max_abs_mean"] = largest_val.round(4)

abs_col_means = np.abs(X_test_scaled).mean(axis=0)
max_idx = np.argmax(abs_col_means)
largest_col_name = X_test.columns[max_idx]
largest_val = abs_col_means[max_idx]
part_b_checks_file["test_scaled_max_abs_mean_column"] = largest_col_name
part_b_checks_file["test_scaled_max_abs_mean"] = largest_val.round(4)

# Convert part_b_checks_file items to a DataFrame and export
df = pd.DataFrame(list(part_b_checks_file.items()), columns=["metric", "value"])
df.to_csv("outputs/part_b_checks.csv", index=False)
print("outputs/part_b_checks.csv generated successfully...")

##########################################################################
#Task 3: Baseline. Predict the mean of y_train for every validation row and record its metrics.
##########################################################################
# Compute the training target mean
train_mean_val = y_train.mean()
# Create predictions: a constant array matching y_val length
y_pred_val = np.full_like(y_val, fill_value=train_mean_val)
# Record and evaluate metrics
baseline_metrics = {
    "model": "mean_baseline",
    "split": "val",
    "MAE": round(mean_absolute_error(y_val, y_pred_val), 4),
    "MSE": round(mean_squared_error(y_val, y_pred_val), 4),
    "RMSE": np.sqrt(mean_squared_error(y_val, y_pred_val)).round(4),
    "R2": round(r2_score(
        y_val, y_pred_val
    ), 4), 
}
baseline_metrics_df = pd.DataFrame([baseline_metrics])
#print(baseline_metrics_df)

##########################################################################
#Task 4: Tree-depth diagnostic. For DecisionTreeRegressor(max_depth=d, random_state=42) with d 
# in 1, 2, 3, 4, 6, 8, 12 and no limit, fit on the training rows and write outputs/part_b_tree_depth.csv 
# with columns max_depth (write unlimited for no limit), train_rmse, val_rmse. 
# The depth with the lowest val_rmse (ties → smaller depth) is the decision-tree depth used in task 5.
def tree_depth_diagnostic(d, X_train_scaled, y_train, X_val_scaled, y_val):
    model = DecisionTreeRegressor(max_depth=d, random_state=42)
    model.fit(X_train_scaled, y_train)
    #predict for training dataset
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    depth = ""
    if (d == None):
        depth = "unlimited"
    else:
        depth = str(d)

    return {
        "max_depth":  depth,
        #"train_mae": round(mean_absolute_error(y_train, y_pred_train), 4),
        "train_mse": round(mean_squared_error(y_train, y_pred_train), 4),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)).round(4),
        "train_r2": round(r2_score(
            y_train, y_pred_train
        ), 4),
        #"val_mae": round(mean_absolute_error(y_val, y_pred_val), 4),
        "val_mse": round(mean_squared_error(y_val, y_pred_val), 4),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_pred_val)).round(4),
        "val_r2": round(r2_score(
            y_val, y_pred_val
        ), 4), 
    }

depths = [1, 2, 3, 4, 6, 8, 12, None]
diognastic_df = pd.DataFrame()
for d in depths:
    result = tree_depth_diagnostic(d, X_train_scaled, y_train, X_val_scaled, y_val)
    result_df = pd.DataFrame([result])
    diognastic_df = pd.concat([diognastic_df, result_df], ignore_index=True)
    
#sorted_df = diognastic_df.sort_values(by="val_rmse", ascending=True)
#print(diognastic_df)
diognastic_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
print("outputs/part_b_tree_depth.csv generated successfully...")

######################################################
#Task 5. Four candidate models, each fitted on the training rows and scored on the validation 
# rows with MAE, MSE, RMSE and R²: LinearRegression inside a Pipeline of ("scaler", StandardScaler()) 
# then ("model", LinearRegression()); the decision tree from task 4; 
# RandomForestRegressor(n_estimators=100, random_state=42); 
# GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42). 
# Write outputs/part_b_metrics.csv with columns model, split, MAE, MSE, RMSE, R2 and rows 
# named mean_baseline, linear_regression, decision_tree, random_forest, 
# gradient_boosting (all split = val), plus the final-test row from task 7. 
# RMSE must equal the square root of MSE (±0.001).
def get_metrics(model, split, actual, predicted):
    metrics = {
            "model": model,
            "split": split,
            "MAE": round(mean_absolute_error(actual, predicted), 4),
            "MSE": round(mean_squared_error(actual, predicted), 4),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)).round(4),
            "R2": round(r2_score(
                actual, predicted
            ), 4), 
        }
    metrics_df = pd.DataFrame([metrics])
    return metrics_df

conslidated_df = pd.DataFrame()
#baseline mean
conslidated_df = pd.concat([conslidated_df, baseline_metrics_df], ignore_index=True)
#LinearRegression
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()), 
    ("model", LinearRegression())
    ])
pipeline_lr.fit(X_train_scaled, y_train)
y_pred_val = pipeline_lr.predict(X_val_scaled)
result_df = get_metrics("linear_regression", "val", y_val, y_pred_val)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)
#DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred_val = model.predict(X_val_scaled)
result_df = get_metrics("decision_tree", "val", y_val, y_pred_val)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)
#RandomForestRegressor(n_estimators=100, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred_val = model.predict(X_val_scaled)
result_df = get_metrics("random_forest", "val", y_val, y_pred_val)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)
#GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_gbr.fit(X_train_scaled, y_train)
y_pred_val = model_gbr.predict(X_val_scaled)
result_df = get_metrics("gradient_boosting", "val", y_val, y_pred_val)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)

###########################################################
#Task 6: Learning-rate experiment. 
# Fit GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42) 
# for lr in 0.01, 0.1, 1.0 and write outputs/part_b_gb_learning_rate.csv (learning_rate, val_rmse, val_r2).
def get_lr_metrics(lr, actual, predicted):
    metrics = {
            "learning_rate": lr,
            #"MAE": round(mean_absolute_error(actual, predicted), 4),
            #"MSE": round(mean_squared_error(actual, predicted), 4),
            "val_rmse": np.sqrt(mean_squared_error(actual, predicted)).round(4),
            "val_r2": round(r2_score(
                actual, predicted
            ), 4), 
        }
    metrics_df = pd.DataFrame([metrics])
    return metrics_df

def learning_rate_exp(lr, X_train_scaled, y_train, X_val_scaled, y_val):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred_val = model.predict(X_val_scaled)
    result_df = get_lr_metrics(lr, y_val, y_pred_val)
    return result_df

rates = [0.01, 0.1, 1.0]
learning_df = pd.DataFrame()
for lr in rates:
    result_df = learning_rate_exp(lr, X_train_scaled, y_train, X_val_scaled, y_val)
    learning_df = pd.concat([learning_df, result_df], ignore_index=True)

#print(learning_df)
learning_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
print("outputs/part_b_gb_learning_rate.csv generated successfully...")

######################################################
#Task 7: Cross-validated comparison and final model. 
# Rank the four candidates by validation RMSE. For the two best, run 5-fold cross-validation with 
# cross_val_score on the 1,020 training-plus-validation rows, using one 
# KFold(n_splits=5, shuffle=True, random_state=42) object shared by both models 
# (default scoring, which is R²). Write outputs/part_b_cv.csv (model, mean_r2, std_r2). 
# The final model is the candidate with the lowest validation RMSE; score it once on the test rows 
# (already fitted on the training rows) and add the row model = <its name>, split = test to part_b_metrics.csv

# Ranking based on rmse_val score the top two models are as given below
#model,             split,  MAE,    MSE,    RMSE,   R2
#linear_regression, val,    3.9222, 23.9235,4.8912, 0.9313
#gradient_boosting, val,    3.9152, 24.1686,4.9162, 0.9306
kf = KFold(n_splits=5, shuffle=True, random_state=42)
#1,020 training-plus-validation rows
X_train_val = np.concat((X_train_scaled, X_val_scaled))
y_train_val = pd.concat([y_train, y_val], ignore_index=True)
#print("Len: ", len(X_train_val))
cv_df = pd.DataFrame()
#LinearRegression
cv_score = cross_val_score(
    LinearRegression(),
    X_train_val, y_train_val, cv=kf
)
cv_data = {
    "model": "linear_regression",
    "mean_r2": round(cv_score.mean(), 4),
    "std_r2": round(cv_score.std(), 4) 
}
tmp_df = pd.DataFrame([cv_data])
cv_df = pd.concat([cv_df, tmp_df], ignore_index=True)
#GradientBoostingRegressor
cv_score = cross_val_score(
    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    X_train_val, y_train_val, cv=kf
)
cv_data = {
    "model": "gradient_boosting",
    "mean_r2": round(cv_score.mean(), 4),
    "std_r2": round(cv_score.std(), 4) 
}
tmp_df = pd.DataFrame([cv_data])
cv_df = pd.concat([cv_df, tmp_df], ignore_index=True)
#print(cv_df)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)
print("outputs/part_b_cv.csv generated successfully...")

# The final model is the candidate with the lowest validation RMSE; score it once on the test rows 
# (already fitted on the training rows) and add the row model = <its name>, split = test to part_b_metrics.csv
#Top Model: LinearRegression on test dataset
y_pred_test = pipeline_lr.predict(X_test_scaled)
result_df = get_metrics("linear_regression", "test", y_test, y_pred_test)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)
#Second Model: Gradient boosting on test dataset
y_pred_test = model_gbr.predict(X_test_scaled)
result_df = get_metrics("gradient_boosting", "test", y_test, y_pred_test)
conslidated_df = pd.concat([conslidated_df, result_df], ignore_index=True)
#print(conslidated_df)
conslidated_df.to_csv("outputs/part_b_metrics.csv", index=False)
print("outputs/part_b_metrics.csv generated successfully...")