import numpy as np
import pandas as pd
import joblib

from src.shipment_prep import print_stats
from src.shipment_prep import clean_shipments
from src.shipment_prep import FEATURES
from src.shipment_prep import NOISE_COLUMNS

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


#########################################################################
# Task 5: Package the model (package_model.py). Read outputs/part_c_final_config.csv, rebuild Part C's 
# stratified training/validation/test split with the same settings, fit the final Pipeline on the 
# training rows, and save with joblib: models/delay_model.pkl; models/delay_threshold.csv 
# (one column threshold); models/feature_order.csv (one column feature, the 17 names in FEATURES order); 
# and models/impute_values.csv (columns column, median — the three medians from clean_shipments). 
# Then reload the pickle with joblib.load, predict probabilities for the 180 test rows with both the 
# reloaded and the in-memory pipeline, and write outputs/part_d_reload_check.csv (metric,value) with 
# test_rows_scored, reload_max_abs_probability_diff and threshold.

#load csv file and build dataframe
config_df = pd.read_csv("outputs/part_c_final_config.csv")
#print(config_df)

#load csv file and build dataframe
data_df = pd.read_csv("data/modelling_table.csv")
X = data_df[FEATURES]
y = data_df["is_delayed"]

#Stratified split. Repeat Part B's two-step split (test_size=0.3 then test_size=0.5, random_state=42 both times) 
# with stratify on the target in both calls. Expected sizes 840 / 180 / 180 with 195 / 42 / 41 delayed parcels.
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print(f"X_train: {len(X_train)} X_val: {len(X_val)}, X_test: {len(X_test)}")
print(f"y_train: {(y_train == 1).sum()} y_val: {(y_val == 1).sum()}, y_test: {(y_test == 1).sum()}")

#fit the train rows on final model pipeline
k_val = config_df["selector_k"].iloc[0]
est_val = config_df["n_estimators"].iloc[0]
depth_val = config_df["max_depth"].iloc[0]
leaf_val = config_df["min_samples_leaf"].iloc[0]
threshold_val = config_df["threshold"].iloc[0]

print(f"k_val: {k_val}, est_val: {est_val}, depth_val: {depth_val}, leaf_val: {leaf_val}")
pipeline_rf_final = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=k_val)), 
    ("model", RandomForestClassifier(n_estimators=est_val, criterion="gini", max_depth=depth_val, min_samples_leaf=leaf_val, random_state=42))
    ])
pipeline_rf_final.fit(X_train, y_train)

# Save the model to disk
joblib.dump(pipeline_rf_final, "models/delay_model.pkl")
print("Model successfully saved to models/delay_model.pkl")

output_df = pd.DataFrame([{"threshold": threshold_val}])
#print(output_df)
#write to csv file
output_df.to_csv("models/delay_threshold.csv", index=False)
print("models/delay_threshold.csv generated successfully...")

output_df = pd.DataFrame([{"feature": ",".join(FEATURES)}])
#print(output_df)
#write to csv file
output_df.to_csv("models/feature_order.csv", index=False)
print("models/feature_order.csv generated successfully...")

#read raw shipments
df_raw_shipments = pd.read_csv('data/raw_shipments.csv')
#call clean_shipment to clean up raw data
clean_df, median_values = clean_shipments(df_raw_shipments)
output_df = pd.DataFrame([{"column": clean_df.shape[1], 
                           "distance_km_median": median_values["distance_km"], 
                           "parcel_weight_kg_median": median_values["parcel_weight_kg"], 
                           "declared_value_inr_median": median_values["declared_value_inr"]}])
#print(output_df)
#write to csv file
output_df.to_csv("models/impute_values.csv", index=False)
print("models/impute_values.csv generated successfully...")

# reload the pickle with joblib.load, predict probabilities for the 180 test rows with both the 
# reloaded and the in-memory pipeline, and write outputs/part_d_reload_check.csv (metric,value) with 
# test_rows_scored, reload_max_abs_probability_diff and threshold.
#--------------------------------
# Load the model from disk
loaded_model = joblib.load("models/delay_model.pkl")

y_prob_test_loaded = loaded_model.predict_proba(X_test)
y_prob_test_memory = pipeline_rf_final.predict_proba(X_test)

#get absolute difference between the two probabilities
diff = np.abs(y_prob_test_loaded - y_prob_test_memory)
#print("max diff: ",diff[:, 1].max())

choosen_t = threshold_val
# add details
final_list = []
final_list.append({"metric": "test_rows_scored", "value": X_test.shape[0]})
final_list.append({"metric": "reload_max_abs_probability_diff", "value": diff[:, 1].max()})
final_list.append({"metric": "threshold", "value": choosen_t})

df_loaded = pd.DataFrame(final_list)
#write to csv file
df_loaded.to_csv("outputs/part_d_reload_check.csv", index=False)
print("outputs/part_d_reload_check.csv generated successfully...")