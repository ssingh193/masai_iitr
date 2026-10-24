import numpy as np
import pandas as pd

# Show all columns
pd.set_option('display.max_columns', None)


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def print_stats(df):
    print("***********************************")
    print("*******[DataFrame Statistics]******")
    print("***********************************")
    print("Total Records: ", len(df))
    print("Null Records: ", df.isnull().sum().sum())
    print("Duplicate records: ", df.duplicated().sum())
    print("Size: ", df.size)
    print("Shape: ", df.shape)
    print("*********[Describe]****************")
    print(df.describe())
    print("*************[Head]****************")
    print(df.head())
    print("***********[Info]******************")
    print(df.info())
    print("***********************************")

def clean_shipments(raw, medians=None):
    print("*********[clean_shipments]****************")
    print("Initial count: ", len(raw))
    print("duplicate records: ", raw.duplicated().sum())
    #2.i Remove Duplicate Rows
    clean_df = raw
    if (raw.duplicated().sum() > 0):
        clean_df = raw.drop_duplicates()
        print("Dropped duplicates, count: ", len(clean_df))

    #2.ii service_type: strip leading and trailing spaces and convert to title case
    clean_df["service_type"] = clean_df["service_type"].astype(str).str.strip().str.title()

    #2.iii payment_mode: replace missing value with 'Unknown'
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
    
    #2.iv declared_value: remove the rupee sign and the thousands commas, convert to a number, 
    # store it in a new column declared_value_inr, and drop the text column
    clean_df["declared_value_inr"] = clean_df["declared_value"].astype(str).replace('₹', '', regex=False).str.replace(",", "", regex=False).str.strip()
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].str.replace(r'[^\d.]', '', regex=True)
    clean_df["declared_value_inr"] = pd.to_numeric(clean_df["declared_value_inr"], errors='coerce')
    #clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(0.0)
    # drop declared_value column
    clean_df.drop(columns=["declared_value"], inplace=True)

    #2.v dispatch_time: convert to a datetime;
    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"], errors='coerce')

    #2.vi parcel_weight_kg: any value above 100 is a scanner-fault sentinel 
    # (the network accepts parcels up to 40 kg) — treat it as missing;
    clean_df["parcel_weight_kg"] = np.where(clean_df["parcel_weight_kg"] > 100, np.nan, clean_df["parcel_weight_kg"])
    #filtered_df = clean_df[(clean_df["parcel_weight_kg"] > 100)]

    #2.vii fill missing distance_km, parcel_weight_kg and declared_value_inr with that column's median. 
    # If medians is None, compute the three medians from the non-missing values left after steps 1–6; 
    # otherwise use the dictionary passed in. The returned medians is a dictionary with exactly the keys distance_km, parcel_weight_kg, declared_value_inr and float values.
    distance_km_median = clean_df["distance_km"].median().round(4)
    if(distance_km_median == np.nan or distance_km_median == None):
        distance_km_median = np.median(clean_df["distance_km"].dropna()).round(4)
    if(distance_km_median == np.nan or distance_km_median == None):
            distance_km_median = medians.get("distance_km", np.nan).round(4)

    parcel_weight_kg_median = clean_df["parcel_weight_kg"].median().round(4)
    if(parcel_weight_kg_median == np.nan or parcel_weight_kg_median == None):
            parcel_weight_kg_median = np.median(clean_df["parcel_weight_kg"].dropna()).round(4)
    if(parcel_weight_kg_median == np.nan or parcel_weight_kg_median == None):
                parcel_weight_kg_median = medians.get("parcel_weight_kg", np.nan).round(4)

    declared_value_inr_median = clean_df["declared_value_inr"].median().round(4)
    if(declared_value_inr_median == np.nan or declared_value_inr_median == None):
            declared_value_inr_median = np.median(clean_df["declared_value_inr"].dropna()).round(4)
    if(declared_value_inr_median == np.nan or declared_value_inr_median == None):
                    declared_value_inr_median = medians.get("declared_value_inr", np.nan).round(4)

    median_values = {"distance_km": distance_km_median, "parcel_weight_kg": parcel_weight_kg_median, "declared_value_inr": declared_value_inr_median}
    print(median_values)
    clean_df["distance_km"] = clean_df["distance_km"].fillna(distance_km_median)
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(parcel_weight_kg_median)
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(declared_value_inr_median)
    #print_stats(clean_df)
    #return {"median_values": median_values, "clean_df": clean_df}
    return (clean_df, median_values)


def build_features(clean, hubs):
    print("*********[build_features]****************")
    # It returns a DataFrame with the hub columns attached by a left merge on hub_id
    clean = pd.merge(clean, hubs, on="hub_id", how="left")
    #new columns: promised_hours (from PROMISED_HOURS, integer)
    clean["promised_hours"] = np.where(
        clean["service_type"] == "Express", PROMISED_HOURS.get("Express"),
        np.where(clean["service_type"] == "Standard", PROMISED_HOURS.get("Standard"), PROMISED_HOURS.get("Economy"))
        )
    #is_cod (1 when payment_mode is COD, else 0 — Unknown gives 0),
    clean["is_cod"] = np.where(clean["payment_mode"] == "COD", 1, 0)
    #is_weekend (1 when the dispatch day is Saturday or Sunday), 
    clean["is_weekend"] = np.where(
            clean["dispatch_time"].dt.day_name() == "Saturday", 1,
            np.where(clean["dispatch_time"].dt.day_name() == "Sunday", 1, 0)
            )
    # dispatch_hour (the hour of the dispatch timestamp as an integer; the generated data uses hours 6 to 21), 
    clean["dispatch_hour"] = clean["dispatch_time"].dt.hour
    # and log_declared_value (natural log of declared_value_inr). It must not use delivery_hours, so that it also works on parcels that have not been delivered yet.
    clean["log_declared_value"] = np.log(clean["declared_value_inr"]).round(4)
    #print_stats(clean)
    return clean
        
        

