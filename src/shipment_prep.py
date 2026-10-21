
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
    print("*********[Describe]****************")
    print(df.describe())
    print("*************[Head]****************")
    print(df.head())
    print("***********[Info]******************")
    print(df.info())
    print("***********************************")

def clean_shipments(raw, medians=None):
    #print(raw.isnull().sum())
    #print("Null %: ", (raw.isnull().sum() / len(raw)) * 100)
    print("Initial count: ", len(raw))
    print("duplicate records: ", raw.duplicated().sum())
    #2.i Remove Duplicate Rows
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
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(0.0)
    # drop declared_value column
    clean_df.drop(columns=["declared_value"], inplace=True)
    print_stats(clean_df)

    pass


#read hubs.csv
df_hubs = pd.read_csv('../data/hubs.csv')
#print(df_hubs.head())
#print("Total Records: ", len(df_hubs))
#print("********************************\n")

#read new shipments
df_new_shipments = pd.read_csv('../data/new_shipments.csv')
#print(df_new_shipments.head())
#print("Total Records: ", len(df_new_shipments))
#print("********************************\n")

#read raw shipments
df_raw_shipments = pd.read_csv('../data/raw_shipments.csv')
#print(df_raw_shipments.head())
#print("Total Records: ", len(df_raw_shipments))
#print("********************************\n")
#call clean_shipment to clean up raw data
clean_shipments(df_raw_shipments)