###########################################################
Part A — Data Foundation and Delay Evidence
###########################################################
"Data findings"
----------------
(i) which service type is delayed most often and at what rate
Express service type is delayed most often and its delay rate is 0.4897

(ii) how delay rate changes from Tier 1 to Tier 3
Tier 1 has a delay rate of 0.0400 while tier 3 has a delay rate of 0.4400. So there is a significant increase in delays from tier 1 to tier 3.

(iii) how it changes across distance bands
As the distance band increases the delay rate also increases. For deliveries above 600 km the delay rate significantly increases to 0.8523

(iv) the three columns most strongly correlated with delivery_hours
corr(delivery_hours, promised_hours) = 0.8372
corr(delivery_hours, distance_km) = 0.4013
corr(delivery_hours, rain_flag) = 0.1437

(v) one sentence on why declared_value_inr was log-transformed
declared_value_inr was log transformed since it had a wider range with min value 80 and max value 12967. Log transformation will help during modeling based on ML algorithms which are impacted by distance.

declared_value_mean_inr,1533.0542
log_declared_value_mean,7.0277
declared_value_median_inr,1121.5
log_declared_value_median,7.0224


###########################################################
Part B — Delivery-Hours Regression
###########################################################