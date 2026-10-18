
import numpy as np
import pandas as pd


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

#read hubs.csv
df_hubs = pd.read_csv('../data/hubs.csv')
print(df_hubs.head())
#read new shipments
df_new_shipments = pd.read_csv('../data/new_shipments.csv')
print(df_new_shipments.head())
#read raw shipments
df_raw_shipments = pd.read_csv('../data/raw_shipments.csv')
print(df_raw_shipments.head())