import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments
from src.shipment_prep import build_features
from src.shipment_prep import print_stats
#from src.shipment_prep import FEATURES

#read hubs.csv
df_hubs = pd.read_csv('data/hubs.csv')
#print(df_hubs.head())
#print("Total Records: ", len(df_hubs))
#print("********************************\n")

#read new shipments
df_new_shipments = pd.read_csv('data/new_shipments.csv')
#print(df_new_shipments.head())
#print("Total Records: ", len(df_new_shipments))
#print("********************************\n")

#read raw shipments
df_raw_shipments = pd.read_csv('data/raw_shipments.csv')
#print(df_raw_shipments.head())
#print("Total Records: ", len(df_raw_shipments))
#print("********************************\n")

#call clean_shipment to clean up raw data
clean_df, median_values = clean_shipments(df_raw_shipments)
clean_df = build_features(clean_df, df_hubs)

#add the target is_delayed (1 when delivery_hours > promised_hours, else 0); 
clean_df["is_delayed"] = np.where((clean_df["delivery_hours"] > clean_df["promised_hours"]), 1, 0)
#print_stats(clean_df)

#write the result to data/modelling_table.csv (all columns, one row per unique shipment)
clean_df.to_csv("data/modelling_table.csv", index=False)
print("data/modelling_table.csv generated successfully...")

# Checks file. The same script writes outputs/part_a_checks.csv with two columns, metric and value, 
# containing the metrics listed in the reference table below, computed by code from the data — never typed in. 
# "Before" counts are measured after removing duplicates and before any filling; 
# missing_parcel_weight_before does not include the sentinel rows. 
# service_type_spellings_before and service_type_spellings_after are the number of distinct service_type strings 
# in the table (after removing duplicates) before and after the strip-and-title-case step, for example with nunique(). 
# nulls_in_features_and_target counts missing cells over the 17 FEATURES columns plus delivery_hours and is_delayed. 
# The four declared-value metrics (declared_value_mean_inr, declared_value_median_inr, log_declared_value_mean, log_declared_value_median) 
# are computed on the median-filled declared_value_inr column of the modelling table (and its natural log), 
# not on the non-missing values only. 
checks_file = {}
checks_file["raw_rows"] = len(df_raw_shipments)
checks_file["duplicate_rows_removed"] = df_raw_shipments.duplicated().sum()
#remove duplicates
df_raw_shipments = df_raw_shipments.drop_duplicates()

checks_file["rows_after_dedup"] = len(df_raw_shipments)
checks_file["missing_distance_km_before"] = df_raw_shipments["distance_km"].isnull().sum()
checks_file["missing_parcel_weight_before"] = df_raw_shipments["parcel_weight_kg"].isnull().sum()
checks_file["sentinel_weight_rows"] = df_raw_shipments[(df_raw_shipments["parcel_weight_kg"] > 100)].value_counts().count()
checks_file["missing_declared_value_before"] = df_raw_shipments["declared_value"].isnull().sum()
checks_file["missing_payment_mode_before"] = df_raw_shipments["payment_mode"].isnull().sum()
checks_file["service_type_spellings_before"] = df_raw_shipments["service_type"].nunique()
checks_file["service_type_spellings_after"] = clean_df["service_type"].nunique()
checks_file["declared_value_mean_inr"] = clean_df["declared_value_inr"].mean().round(4)
checks_file["log_declared_value_mean"] = clean_df["log_declared_value"].mean().round(4)
checks_file["median_distance_km"] = clean_df["distance_km"].median().round(4)
checks_file["median_parcel_weight_kg"] = clean_df["parcel_weight_kg"].median().round(4)
checks_file["median_declared_value_inr"] = clean_df["declared_value_inr"].median().round(4)
checks_file["rows_after_hub_merge"] = len(clean_df)
checks_file["nulls_in_features_and_target"] = clean_df.isna().sum().sum()
checks_file["delayed_count"] = clean_df[(clean_df["is_delayed"] > 0)].value_counts().count()
checks_file["delay_rate"] = round(clean_df[(clean_df["is_delayed"] > 0)].value_counts().count() / len(clean_df), 4)

# The hours_* metrics use the 25th and 75th percentiles of delivery_hours and the 1.5 × IQR rule; 
# compute the bounds from the unrounded quartiles and round only when writing the file.
Q1 = np.percentile(clean_df["delivery_hours"], 25)
Q2 = np.percentile(clean_df["delivery_hours"], 50)
Q3 = np.percentile(clean_df["delivery_hours"], 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = clean_df[(clean_df["delivery_hours"] < lower) | (clean_df["delivery_hours"] > upper)].value_counts().count()

checks_file["hours_q1 / hours_q3"] = f"{Q1.round(2)} / {Q3.round(2)}"
checks_file["hours_lower_bound / hours_upper_bound"] = f"{lower.round(2)} / {upper.round(2)}"
checks_file["hours_outlier_count"] = outliers

checks_file["declared_value_median_inr"] = clean_df["declared_value_inr"].median().round(4)
checks_file["log_declared_value_median"] = clean_df["log_declared_value"].median().round(4)

# Convert checks_file items to a DataFrame and export
df = pd.DataFrame(list(checks_file.items()), columns=["Metric", "Value"])
df.to_csv("outputs/part_a_checks.csv", index=False)
print("outputs/part_a_checks.csv generated successfully...")

#Delay evidence table. Write outputs/part_a_delay_by_group.csv with columns 
# dimension, group, shipments, delayed, delay_rate, mean_hours, one row per group, 
# for three dimensions: service_type; hub_tier; and distance_band, where distance_band is distance_km 
# cut into the bins 0-200, 201-400, 401-600, 601+ (edges 0, 200, 400, 600, 2000; right edge inclusive) 
# with those labels. delay_rate is the share of delayed parcels in the group; mean_hours is the 
# mean of delivery_hours
df = clean_df.groupby(["service_type"])[["shipment_id", "is_delayed", "delivery_hours"]].agg(
    {
        "shipment_id": "count",
        "is_delayed": "sum",
        "delivery_hours": "mean"
    }
).round(4).reset_index()
df["dimension"] = "service_type"
df["delay_rate"] = round(df["is_delayed"] / df["shipment_id"], 4)
#rename the columns and rearrange
df = df.rename(
    columns={
        "service_type": "group",
        "shipment_id": "shipments",
        "is_delayed": "delayed",
        "delivery_hours": "mean_hours",
    }
)[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]

#print(df)

df_hub_tier = clean_df.groupby(["hub_tier"])[["shipment_id", "is_delayed", "delivery_hours"]].agg(
    {
        "shipment_id": "count",
        "is_delayed": "sum",
        "delivery_hours": "mean"
    }
).round(4).reset_index()
df_hub_tier["dimension"] = "hub_tier"
df_hub_tier["delay_rate"] = round(df_hub_tier["is_delayed"] / df_hub_tier["shipment_id"], 4)
#rename the columns and rearrange
df_hub_tier = df_hub_tier.rename(
    columns={
        "hub_tier": "group",
        "shipment_id": "shipments",
        "is_delayed": "delayed",
        "delivery_hours": "mean_hours",
    }
)[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]

#distance_band, where distance_band is distance_km 
# cut into the bins 0-200, 201-400, 401-600, 601+ (edges 0, 200, 400, 600, 2000; right edge inclusive) 
# with those labels.
bins = [0, 200, 400, 600, 2000]
labels = ["0-200", "201-400", "401-600", "601+"]
clean_df["distance_band"] = pd.cut(
    clean_df["distance_km"], bins=bins, labels=labels, right=True
)

df_distance_band = clean_df.groupby(["distance_band"])[["shipment_id", "is_delayed", "delivery_hours"]].agg(
    {
        "shipment_id": "count",
        "is_delayed": "sum",
        "delivery_hours": "mean"
    }
).round(4).reset_index()
df_distance_band["dimension"] = "distance_band"
df_distance_band["delay_rate"] = round(df_distance_band["is_delayed"] / df_distance_band["shipment_id"], 4)
#rename the columns and rearrange
df_distance_band = df_distance_band.rename(
    columns={
        "distance_band": "group",
        "shipment_id": "shipments",
        "is_delayed": "delayed",
        "delivery_hours": "mean_hours",
    }
)[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]

#combine all the dataframes
df_combined = pd.concat([df, df_hub_tier, df_distance_band], ignore_index=True)
#print(df_combined)
# SAve dataframe to csv file
df_combined.to_csv("outputs/part_a_delay_by_group.csv", index=False)
print("outputs/part_a_delay_by_group.csv generated successfully...")

#Correlations. Write outputs/part_a_correlations.csv: 
# the correlation matrix of the numeric columns of the modelling table (row labels kept as the first column).
#Correlation spot-checks (±0.01): 
# corr(delivery_hours, promised_hours) = 0.837; 
# corr(delivery_hours, distance_km) = 0.401; 
# corr(is_delayed, distance_km) = 0.546.

correlation_data = {}
correlation_data["corr(delivery_hours, promised_hours)"] = clean_df["delivery_hours"].corr(clean_df["promised_hours"]).round(4)
correlation_data["corr(delivery_hours, distance_km)"] = clean_df["delivery_hours"].corr(clean_df["distance_km"]).round(4)
correlation_data["corr(is_delayed, distance_km)"] = clean_df["is_delayed"].corr(clean_df["distance_km"]).round(4)
correlation_data["corr(delivery_hours, parcel_weight_kg)"] = clean_df["delivery_hours"].corr(clean_df["parcel_weight_kg"]).round(4)
correlation_data["corr(is_delayed, parcel_weight_kg)"] = clean_df["is_delayed"].corr(clean_df["parcel_weight_kg"]).round(4)
correlation_data["corr(is_delayed, is_weekend)"] = clean_df["is_delayed"].corr(clean_df["is_weekend"]).round(4)
correlation_data["corr(is_delayed, dispatch_hour)"] = clean_df["is_delayed"].corr(clean_df["dispatch_hour"]).round(4)
correlation_data["corr(delivery_hours, dispatch_hour)"] = clean_df["delivery_hours"].corr(clean_df["dispatch_hour"]).round(4)
correlation_data["corr(is_delayed, rain_flag)"] = clean_df["is_delayed"].corr(clean_df["rain_flag"]).round(4)
correlation_data["corr(delivery_hours, rain_flag)"] = clean_df["delivery_hours"].corr(clean_df["rain_flag"]).round(4)
correlation_data["corr(delivery_hours, is_delayed)"] = clean_df["delivery_hours"].corr(clean_df["is_delayed"]).round(4)
correlation_data["corr(delivery_hours, is_weekend)"] = clean_df["delivery_hours"].corr(clean_df["is_weekend"]).round(4)


# Convert checks_file items to a DataFrame and export
df = pd.DataFrame(list(correlation_data.items()), columns=["Correlation", "Value"])
df.to_csv("outputs/part_a_correlations.csv", index=False)
print("outputs/part_a_correlations.csv generated successfully...")

############################################################
## TESTING
############################################################
#clean_df2, mean_val = clean_shipments(df_new_shipments, median_values)
#clean_df2 = build_features(clean_df2, df_hubs)
#print_stats(clean_df2)