###########################################################
Part A — Data Foundation and Delay Evidence
###########################################################
----------------
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
-----------------------------
"Model results (regression)"
-----------------------------
1. baseline R² = -0.0212
2. Which model won on validation and its RMSE in hours? 
   LinearRegression with RMSE value as 4.8912

3. Which depths are high-bias and which are high-variance (use the gap between train_rmse and val_rmse)?
   depth 1 is high-bias with train_rmse=11.6334 and val_rmse=12.1324 both on the higher side showing no generalization at all (underfitting).
   depth unlimited (max_depth=None) has high variance with 0.0 as train_rmse while 7.7871 as val_rmse (overfitting).

4. Which learning rate was best and why the other two fail (too slow versus overshooting)?
   0.1 learning rate was best with lowest val_rmse of 4.9162
   0.01 learning rate had the val_rmse of 10.0018 showing it was too slow
   1.0 learning rate has the val_rmse of 7.2823 showing it overshoot the best learning rate value.

5. If the cross-validation ranking disagrees with the validation ranking, which model you would ship and why.
   Yes, cross-validation when run on the top two models with validation data set shows thedifferent results where 
   LinearRegression model gave lower mean_r2 value than the GradientBoostingRegressor model.
   Below are the CV results on validation data set for the top two models.
   ------------------------------------------------------------
   model,               mean_r2,    std_r2
   linear_regression,   0.9261,     0.0048
   gradient_boosting,   0.9352,     0.0045
   ------------------------------------------------------------
   But i would still consider the winner as LinearRegression model as its performance on test data set is better than GradientBoostingRegressor. So final winner is LinearRegression model.
   ############################################################
   #Final Model result on Test dataset
   #model,             split, MAE,     MSE,     RMSE,    R2
   #linear_regression, test,  3.5671,  20.4624, 4.5235,  0.9493
   ############################################################


###########################################################
Part C — Delay Classifier and Alert Threshold
###########################################################