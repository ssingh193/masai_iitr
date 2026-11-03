import numpy as np
import pandas as pd
import joblib

from src.shipment_prep import print_stats
from src.shipment_prep import clean_shipments
from src.shipment_prep import build_features
from src.shipment_prep import FEATURES
from src.shipment_prep import NOISE_COLUMNS




##########################################################################
# Task 6: Scoring script (predict_delay.py). Load data/new_shipments.csv, data/hubs.csv, the four files 
# in models/ and outputs/hub_segments.csv; clean the new parcels with your own clean_shipments, 
# passing the stored medians; attach features with your own build_features; arrange the columns in the 
# order of feature_order.csv; compute delay_probability (positive class, rounded to 4 decimals); 
# set flagged = 1 when the probability is ≥ the stored threshold; join the hub segment; and set action 
# to Escalate to hub manager when the parcel is flagged and its hub's segment is Escalate, Proactive 
# customer call when it is flagged otherwise, and No action when it is not flagged. 
# Write outputs/new_shipment_actions.csv with columns shipment_id, hub_id, segment, delay_probability, 
# flagged, action, one row per new parcel, in file order. Running the script twice must produce 
# byte-identical output.

#load csv file and build dataframe
new_shipments_df = pd.read_csv("data/new_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")
segments_df = pd.read_csv("outputs/hub_segments.csv")
#3 files in models
threshold_df = pd.read_csv("models/delay_threshold.csv")
feature_df = pd.read_csv("models/feature_order.csv")
impute_df = pd.read_csv("models/impute_values.csv")

print("new_shipments_df: ", new_shipments_df.shape)
print("hubs_df: ", hubs_df.shape)
print("segments_df: ", segments_df.shape)
print("threshold_df: ", threshold_df.shape)
print("feature_df: ", feature_df.shape)
print("impute_df: ", impute_df.shape)
#print_stats(new_shipments_df)

# clean new_shipments_df by passing median from impute_values.csv
median_values = {
    "distance_km": impute_df["distance_km_median"].iloc[0], 
    "parcel_weight_kg": impute_df["parcel_weight_kg_median"].iloc[0], 
    "declared_value_inr": impute_df["declared_value_inr_median"].iloc[0]
    }
#print(median_values)
shipment_new_clean_df, returned_mean = clean_shipments(new_shipments_df, median_values)
#print_stats(shipment_new_clean_df)
#attach features with your own build_features; 
shipment_new_clean_df = build_features(shipment_new_clean_df, hubs_df)
# arrange the columns in the order of feature_order.csv;
feature_list = (feature_df["feature"].iloc[0]).split(",")
#print(feature_list)
clean_df = shipment_new_clean_df[feature_list]
#print_stats(clean_df)

# Load the model from disk
loaded_model = joblib.load("models/delay_model.pkl")
#compute delay_probability (positive class, rounded to 4 decimals);
X_new = clean_df
y_new = loaded_model.predict_proba(X_new)
y_new_positive = y_new[:, 1].round(4)
#print("y_new_positive: ", y_new_positive)

# set flagged = 1 when the probability is ≥ the stored threshold;
y_pred = (y_new_positive >= threshold_df["threshold"].iloc[0]).astype(int)
#print("y_pred: ", y_pred)
shipment_new_clean_df["is_delayed"] = y_pred
print_stats(shipment_new_clean_df)

#join the hub segment; and set action to Escalate to hub manager when the parcel is flagged and 
# its hub's segment is Escalate, Proactive customer call when it is flagged otherwise, 
# and No action when it is not flagged.