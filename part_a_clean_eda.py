import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments
from src.shipment_prep import build_features
from src.shipment_prep import print_stats

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
result = clean_shipments(df_raw_shipments)
clean_df = build_features(result.get("clean_df"), df_hubs)

#add the target is_delayed (1 when delivery_hours > promised_hours, else 0); 
clean_df["is_delayed"] = np.where((clean_df["delivery_hours"] > clean_df["promised_hours"]), 1, 0)
#print_stats(clean_df)

#write the result to data/modelling_table.csv (all columns, one row per unique shipment)
clean_df.to_csv("data/modelling_table.csv", index=False)
print("data/modelling_table.csv generated successfully...")