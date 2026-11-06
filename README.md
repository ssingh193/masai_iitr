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
---------------------------------
"Model results (classification)"
---------------------------------
1. the blind baseline's accuracy and why it is useless?
   The blind baseline's accuracy is 0.7667. This is useless since the baseline accuracy is calculated assuming there are no delayed parcels. The predictions are assumed to be no delays and using that the accuracy is calculated.

2. the noise columns' combined share of the forest's importance and whether all five fall outside the top 7 by F-score?
   Noise columns combined score of forests importance = 0.2245
   Yes, all five noise columns are outside the top 7 selected columns. Please refer below for the score and rank.
   -----------------------------------------------------------------------------
    feature	               f_score	   f_rank	   rf_importance	   is_noise
   ----------------------------------------------------------------------------- 
    label_print_batch	   4.5847	   8	         0.046	            TRUE
    tape_roll_no	         1.9052	   12	         0.0476	         TRUE
    sticker_serial_mod	   1.2776	   14	         0.0499	         TRUE
    shelf_slot_no	         0.1393	   16	         0.0436	         TRUE
    scanner_build_no	      0.023	      17	         0.0374	         TRUE
    
3. what the k table shows?
   k table shows that the validation f1 score increases with k value and reaches a max when k=7 and then drops and later increases again for k=12 and then drops again.
   --------------
   k,    val_f1
   --------------
   5,    0.6988
   7,    0.8333
   10,   0.7857
   12,   0.8049
   17,   0.7848

4. whether tuning beat the untuned forest on validation? and what that says about tuning on a 180-row validation set?
   No, tunning did not beat the untunned forest model on validation data. Both accuracy and f1 score was higher for the untunned forest model. This means that tunning helps primarily on the new unseen data and may give us better results as compared to validation or training data.

5. the chosen threshold and the validation precision/recall trade-off behind it?
   Chosen threshold is 0.4. As the threshold increases from 0.1 to 0.4 precision increases to 0.7347 and recall decreases to 0.8571 showing a trade-off between the both.
   
6. the test precision, recall and F1, comparing test recall with validation recall honestly (the test set has only 41 delayed parcels)?
   Comparing the test precision and recall with validation precision and recall for the same 0.4 threshold we notice that there is a drop in precision and recall for test against the validation.

   --------------------------------------------------------------------------------------
            threshold,  TN,   FP,   FN,   TP,   accuracy,   precision,  recall,  f1
   --------------------------------------------------------------------------------------
   test     0.4,        122,  17,   8,    33,   0.8611,     0.66,       0.8049,  0.7253
   val      0.4,        125,  13,   6,    36,   0.8944,     0.7347,     0.8571,  0.7912
   

########################################################
Part D — Hub Segments, Packaging and Operations Playbook
########################################################
-------------------------------------------
"How to run", "Operations recommendation", 
"Production plan" and "Limitations"
-------------------------------------------

1. How to run: 
   Step 1: Pre-requisites: 
           a. Ensure python is installed on your machine. 
           b. Setup a IDE of your choice (Recommended VSC)
           c. Optional: Install github plugin on your IDE
           d. Setup a workspace for the given project in your IDE
   Step 2: Clone the github repository 
           a. Use the plugin installed on your IDE for github to download the source code in your workspace.

   Step 3: Check for compatibility
           a. Run `pip freeze` command to check python packages and dependencies against the requirements.txt file provided in the main directory
           b. Execute `pip install -r requirements.txt` to automates the entire installation process.

   Step 4: Regenerate the model
           a. Run `python package_model.py` to regenrate the `models/delay_model.pkl`
           b. This is required to regenrate the delay model effectively after the required packages are installed.

   Step 5: Predict actions
           a. Run `python predict_delay.ph` to analyze and predict actions for new shipments.
           b. Review proposed actions for the operations manager in output/new_shipment_actions.csv

2. Operations recommendation for the regional head

   a. Hubs in the Escalate segment with their delay rate: 
      Below table shows the hubs which are categorized as escalate due to delays in parcel delivery with delay rate above 0.4.

      hub_id  delay_rate   segment     hub_city
      HUB13        0.46  Escalate       Nagpur
      HUB15        0.36  Escalate        Kochi
      HUB17        0.46  Escalate        Patna
      HUB18        0.38  Escalate       Ranchi
      HUB19        0.62  Escalate     Guwahati
      HUB20        0.48  Escalate       Raipur
      HUB21        0.40  Escalate  Bhubaneswar
      HUB23        0.44  Escalate     Siliguri
      HUB24        0.54  Escalate     Agartala

   b. Every hub in tier_vs_segment_mismatch.csv and what that means for how its tier label should be used?
      Every hub in the below table are matched with incorrect tier based on their segments.
      hub tier should be mapped as per segments as following: 
      Tier 1 → Standard, Tier 2 → Watch, Tier 3 → Escalate

      hub_id,  segment,    hub_city,   hub_tier
      HUB07,   Watch,      Kolkata,    1
      HUB13,   Escalate,   Nagpur,     2
      HUB15,   Escalate,   Kochi,      2
      HUB22,   Watch,      Dehradun,   3

   c. the alert threshold
      Parcel are flagged for managere alerts based on the threshold set in models\delay_threshold.csv
      Current model gives action alerts based on threshold value as 0.4

   d. From part_c_final_test.csv — how many of every 10 flagged parcels are really delayed (precision) and how many of every 10 delayed parcels are caught (recall)?

      For every 10 parcels: Roughly 7 parcels are delayed (0.66 precision) while 8 delayed parcels are caught (0.80 recall) 

      threshold,  TN,   FP,   FN,   TP,   accuracy,   precision,  recall,  f1
      0.4,        122,  17,   8,    33,   0.8611,     0.66,       0.8049,  0.7253

   e. How many of the 48 new parcels were flagged and how many of those need a hub-manager escalation?

      How many of the 48 new parcels were flagged:  12
      How many of those need a hub-manager escalation:  10

   f. One or two sentences explaining why you kept K = 3 for the segments, citing your part_d_separation.csv

      Based on the data from part_d_separation.csv, K = 3 is the optimal choice because it delivers the steepest jump in the across-over-within cluster ratio (nearly doubling from 2.89 to 5.39), while maintaining exceptionally well-balanced and highly operational segment sizes (9, 8, and 7). Moving to (K = 4) or (K = 5) yields diminishing statistical returns and fragments the data into tiny, impractical clusters of 2 or 3 items.

      K,    across_over_within,  cluster_sizes
      2,    2.894,               15-9
      3,    5.3928,              9-8-7
      4,    6.3047,              8-7-5-4
      5,    6.8866,              8-7-4-3-2

3. Production plan
   (a) map at least five of these seven production-readiness characteristics to a concrete item in your repository — accuracy, reliability, reproducibility, performance (speed), maintainability, consistency (same preprocessing in training and prediction), monitorability?
   
   1. Accuracy - Test accuracy for the delay model is 0.8611 which is good given recall of 0.8049 and 0.7253 f1 score.
   2. Reliability - Model is well reliable as it performs consitantly and does not fail.
   3. Reproducibility - Model is reproducible which means for the similar type of test data the results are consistant and/or similar.
   4. Pereformance - Executing the model on given test data performs really well.
   5. Consistancy - Model uses the same preprocessing steps for training and prediction with same random state hence the results are consistant.

   (b) name at least three quantities to monitor with a numeric trigger for each (for example the weekly delayed share of delivered parcels, the weekly share of parcels flagged, and recall measured on parcels whose outcome is now known)?

   1. Every week track the percentage of parcels flagged for delays.
   2. Every week track the percentage of parcels flagged for escalation to hub manager.
   3. Every week track the hub with highest parcels flagged for delays.   

   (c) give a retraining schedule and say how the 15–28 September sale window (sale_event_flag) should be handled — compare against the same kind of event, not against ordinary weeks — and what data drift would look like here?

   When retraining a model that incorporates a massive peak event like the September sale window, treating it as standard data will degrade model's accuracy.
   1. Handling the Sale Window in Retraining
   A massive sales event triggers structural shifts in logistics (e.g., massive volume spikes, courier capacity limits etc) that render standard baseline comparisons non-effective.

   Segment and Compare Apple-to-Apple: 
   Recommendation would be to not train or evaluate the model's performance during the September 15–28 window against ordinary weeks. Instead, benchmark the sale window against previous major sale events (e.g., Prime Day, Black Friday, or the last years September sale window).

   Dual-Model Strategy or Flagging:
	Option A (Contextual Flag): Keep the sale_event_flag == 1 in training data, but ensure the model has historical data from multiple past sale events so it can learn the non-linear relationship between high volume and delays.
	Option B (Dedicated Model): Run a dedicated "Peak Season" model routing mechanism. When sale_event_flag == 1, route traffic to a model trained exclusively on peak historical data.

   What Data Drift Looks Like Here??
   1. Volume Spikes per Hub: Some of the regional hub would suddenly receives (5 times) its normal weekly volume.
   2. Proportion of Flagged Segments: The ratio of high-priority express parcels to standard would increase significantly.


4. Limitations
   a. Synthetic data - The model is completely trained and validated on synthetic data and hence the real life example may result in lower accuracy and precision of the model.
   b. Smaller data set - The model is trained on a small data set of 1200 records and tested on 48 test records with just 12 parcels flagged as delayed.
   c. Threshold value tunning is performed on one single validation split of the dataset.