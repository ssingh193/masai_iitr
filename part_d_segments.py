import numpy as np
import pandas as pd

from src.shipment_prep import print_stats
from src.shipment_prep import FEATURES
from src.shipment_prep import NOISE_COLUMNS

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.spatial.distance import pdist, squareform

#load csv file and build dataframe
data_df = pd.read_csv("data/modelling_table.csv")
#print_stats(data_df)

#############################################
#Task 1: Hub profile (part_d_segments.py). 
# From data/modelling_table.csv, group by hub_id and compute shipments (count), mean_delivery_hours, 
# delay_rate (share of delayed parcels), mean_distance_km and mean_n_stops. Write outputs/hub_profile.csv 
# (24 rows, sorted by ascending hub_id; columns hub_id plus those five).
df = data_df.groupby(["hub_id"])[["shipment_id", "delivery_hours", "is_delayed", "distance_km", "n_stops"]].agg(
    {
        "shipment_id": "count",
        "delivery_hours": "mean",
        "is_delayed": "mean",
        "distance_km": "mean",
        "n_stops": "mean"
    }
).round(4).reset_index()
#rename the columns and rearrange
df = df.rename(
    columns={
        "shipment_id": "shipments",
        "delivery_hours": "mean_delivery_hours",
        "is_delayed": "delay_rate",
        "distance_km": "mean_distance_km",
        "n_stops": "mean_n_stops",
    }
)[["hub_id", "shipments", "mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]

#print(df)
#write to csv file
df.to_csv("outputs/hub_profile.csv", index=False)
print("outputs/hub_profile.csv generated successfully...")

#####################################################################
# Task 2: K-Means segmentation. Scale the four profile columns mean_delivery_hours, delay_rate, 
# mean_distance_km, mean_n_stops with StandardScaler (distance-based clustering needs comparable scales) 
# and fit KMeans(n_clusters=3, random_state=42, n_init=10). Fixed order of the hubs: the hub profile is 
# sorted by ascending hub_id (HUB01 first, HUB24 last — the order groupby returns and the order of 
# hub_profile.csv) before it is scaled and clustered, so the 24 hubs always enter StandardScaler and 
# KMeans in that order; the four profile columns may be in any order. K = 3 is fixed by the business 
# — operations runs exactly three playbooks.

features = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])
#scaled_df = pd.DataFrame(scaled_data, columns=features)
#print(scaled_df)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled_data)
df["cluster"] = labels
#print(df)
#print(df["cluster"].value_counts())

######################################################################
# Task 3: Separation evidence. For K in 2, 3, 4, 5 (each KMeans(n_clusters=K, random_state=42, n_init=10) 
# on the same scaled matrix, hubs in ascending hub_id order) compute the across-to-within distance ratio: 
# the mean Euclidean distance over all unordered hub pairs that sit in different clusters divided by the 
# mean over pairs in the same cluster, measured on the scaled matrix (276 pairs). 
# Write outputs/part_d_separation.csv with columns K, across_over_within, cluster_sizes, where cluster_sizes 
# is the cluster sizes in descending order joined by hyphens (for example 9-8-7).

# Precompute the pairwise Euclidean distance matrix (276 unique pairs for 24 hubs)
dist_matrix = squareform(pdist(scaled_data, metric="euclidean"))
n_hubs = len(scaled_data)

results = []

for K in [2, 3, 4, 5]:
    # Fit KMeans
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)    
    # Calculate cluster sizes in descending order joined by hyphens
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
    cluster_sizes_str = "-".join(map(str, sizes))
    
    # Separate distances into same-cluster (within) and different-cluster (across)
    within_dists = []
    across_dists = []

    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            d = dist_matrix[i, j]
            if labels[i] == labels[j]:
                within_dists.append(d)
            else:
                across_dists.append(d)
                
    # Compute the ratio of means
    mean_within = np.mean(within_dists) if within_dists else 0
    mean_across = np.mean(across_dists) if across_dists else 0
    ratio = mean_across / mean_within if mean_within != 0 else 0
    
    results.append({
        "K": K,
        "across_over_within": round(ratio, 4),
        "cluster_sizes": cluster_sizes_str
    })

output_df = pd.DataFrame(results)
#print(output_df)
#write to csv file
output_df.to_csv("outputs/part_d_separation.csv", index=False)
print("outputs/part_d_separation.csv generated successfully...")

######################################################################
# Task 4: Segment names and tier check. Order the three clusters by the mean of their hubs' delay_rate 
# and name them Standard (lowest), Watch (middle) and Escalate (highest). Write outputs/hub_segments.csv 
# (hub_id, segment, hub_city, hub_tier), outputs/segment_profile.csv (segment, hubs, mean_delivery_hours, 
# delay_rate, mean_distance_km, mean_n_stops — means of the hub-level values) and 
# outputs/tier_vs_segment_mismatch.csv (the rows of hub_segments.csv whose segment differs from the one 
# their tier implies: Tier 1 → Standard, Tier 2 → Watch, Tier 3 → Escalate).

# Calculate the mean delay_rate for each cluster to determine rank
mean_delays = df.groupby("cluster")["delay_rate"].mean().sort_values()

# Map the sorted clusters to their new descriptive names
# Index 0 has the lowest delay (Standard), Index 2 has the highest (Escalate)
cluster_mapping = {
    mean_delays.index[0]: "Standard",
    mean_delays.index[1]: "Watch",
    mean_delays.index[2]: "Escalate"
}

# Apply the names to a new column
df["segment"] = df["cluster"].map(cluster_mapping)
#print(df)
#cluster_profiles = df.groupby("segment")[features].mean()
#print(cluster_profiles)

# Merge only the required lookup columns into your main dataframe
df_merged = pd.merge(
    df, 
    data_df[["hub_id", "hub_city", "hub_tier"]], 
    on="hub_id", 
    how="left",
    validate="one_to_many"
)
df_merged = df_merged.drop_duplicates()
hub_segments_df = df_merged[["hub_id", "segment", "hub_city", "hub_tier"]]
#print(hub_segments_df)
#write to csv file
hub_segments_df.to_csv("outputs/hub_segments.csv", index=False)
print("outputs/hub_segments.csv generated successfully...")

#outputs/segment_profile.csv (segment, hubs, mean_delivery_hours, 
# delay_rate, mean_distance_km, mean_n_stops — means of the hub-level values)
segment_profile_df = df_merged[["segment", "hub_city", "mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]]
#write to csv file
segment_profile_df.to_csv("outputs/segment_profile.csv", index=False)
print("outputs/segment_profile.csv generated successfully...")

# outputs/tier_vs_segment_mismatch.csv (the rows of hub_segments.csv whose segment differs from the one 
# their tier implies: Tier 1 → Standard, Tier 2 → Watch, Tier 3 → Escalate).
#print(hub_segments_df)
filtered_df = hub_segments_df[((hub_segments_df["hub_tier"] == 1) & (hub_segments_df["segment"] != "Standard")) |
                              ((hub_segments_df["hub_tier"] == 2) & (hub_segments_df["segment"] != "Watch")) |
                              ((hub_segments_df["hub_tier"] == 3) & (hub_segments_df["segment"] != "Escalate"))
                              ]
#print(filtered_df)
#write to csv file
filtered_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("outputs/tier_vs_segment_mismatch.csv generated successfully...")
