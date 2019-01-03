# High Performance EDA & GLM Modeling with Apache Spark and H2O

This PySpark/H2O based module (optimal_eda_glm.py) is a personal pet project aiming to automate the procedures of exploratory data analysis, numerical and categorical feature engineering and encoding, as well as the optimal linear/logistic regression/GBM model building with randomized hyper-parameter grid search.

Inspired by the method of conditional inference tree in r ctree package (https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf) and smbinning package (https://cran.r-project.org/web/packages/smbinning/smbinning.pdf), I extended the conditional inference tree based variable binning methd to both linear and logistic regression and implemented a slightly different version of hypothesis testing based tree building process.The computationally expensive step of calculating the contingency table based on inital quantile based bins are handled by PySpark in a parallel fashion.The algorithm essentially build a decision tree on a single variable with respect to target variable. The root nodes of the tree will be the unique bins of each variable. 

Here is the algorithm description: with the initial quantile based bins (e.g., 20-quantiles), the algorithm scans and identifies optimal split point based on appropriate hypothesis testing statistic that signals maximum difference in either proprtion (logistic regression) or mean value (linear regression). Specifically, Chi-square testing of independence of sample proportions is applied to the datasets segmented by each initial quantile bin for each numerical variable used by logistic regression. Similarly, Welch T test of two independent sample mean with unknown variance is applied to the datasets segmented by each initial quantile bin for variables used by linear regression. After splitting the dataset into two segments, the two segments will each go through the same divide and conquer procedures using a recursive function until one of the following two conditions is no longer met:

1. P-value of the testing statistics is higher than 0.05 (user defined).
2. The sample size of the bin is smaller than 5% of the total modeling sample size.

Condition 1 aims to ensure the two segments have statistically significant difference that justifies more granular binning. Condition 2 aims to ensure the minimum sample size of the smallest bin is not too small to be practically significant.

After the variable binning, Weight of Evidence (WOE) is calculated and used to replace the original feature value for their corresponding segments in logistic regression. On the other hand, mean value is calculated and used to replace the original feature value for their corresponding segments in linear regression. The procedure will also calculate information value for each variable in logistic regression and R square statistic for each variable in linear regression. This way we are able to compare the univariate predictiveness of both numerical variables and categorical variables on the same basis, since for categorical variables WoE/IV or Mean/R square is calculated directly based on pre-defined categories.

For demonstration purpose, below is an example of using open source dataset from Kaggle Home Credit Default Risk competition to create a logistic regression model for default event prediction. Information value/WoE was calculated based on fixed quantiles (e.g., 20-quantiles). The table below shows the initial EDA result of top feature as measured by information value.

(Please reference the following functions: stack_columns_bucketization, init_woe_iv (logistic regression), init_r_square (linear regression) for initial quantile based binning and EDA procedures using PySpark/Pandas.)

### Preliminary top feature binning result (20-quantiles)


| var_name     | WOE    | lower_bound | higher_bound | overall_IV |
|--------------|--------|-------------|--------------|------------|
| EXT_SOURCE_3 | 1.244  | 0.001       | 0.155        | 0.337      |
| EXT_SOURCE_3 | 0.821  | 0.155       | 0.228        | 0.337      |
| EXT_SOURCE_3 | 0.602  | 0.228       | 0.284        | 0.337      |
| EXT_SOURCE_3 | 0.435  | 0.284       | 0.330        | 0.337      |
| EXT_SOURCE_3 | 0.210  | 0.330       | 0.371        | 0.337      |
| EXT_SOURCE_3 | 0.155  | 0.371       | 0.408        | 0.337      |
| EXT_SOURCE_3 | 0.028  | 0.408       | 0.442        | 0.337      |
| EXT_SOURCE_3 | -0.031 | 0.442       | 0.476        | 0.337      |
| EXT_SOURCE_3 | -0.206 | 0.476       | 0.506        | 0.337      |
| EXT_SOURCE_3 | -0.361 | 0.506       | 0.535        | 0.337      |
| EXT_SOURCE_3 | -0.366 | 0.535       | 0.564        | 0.337      |
| EXT_SOURCE_3 | -0.523 | 0.564       | 0.592        | 0.337      |
| EXT_SOURCE_3 | -0.487 | 0.592       | 0.618        | 0.337      |
| EXT_SOURCE_3 | -0.634 | 0.618       | 0.643        | 0.337      |
| EXT_SOURCE_3 | -0.664 | 0.643       | 0.669        | 0.337      |
| EXT_SOURCE_3 | -0.701 | 0.669       | 0.694        | 0.337      |
| EXT_SOURCE_3 | -0.818 | 0.694       | 0.719        | 0.337      |
| EXT_SOURCE_3 | -0.926 | 0.719       | 0.749        | 0.337      |
| EXT_SOURCE_3 | -0.965 | 0.749       | 0.786        | 0.337      |
| EXT_SOURCE_3 | 0.156  | NULL        | NULL         | 0.337      |


However, 20 bins is quite excessive from scorecard building perspective. Meanwhile, some neighboring bins do not have significantly different level of default risk (as measured by probability of default). 

After applying the recursive bin partitioning based on Chi-Square testing mentioned above to refine and simplify the segments binning, we reduced the total number of bins for EXT_SOURCE_3 while retaining majority of information value, as shown below:  

(please reference update_iv_with_new_bin and recursive_var_bin function for recursive bin optimization for logistic regression, update_r_square_with_new_bin and LR_recursive_var_bin function for recursive bin optimization for linear regression)

### After Chi-squared testing based recursive partitioning (8 bins)

| var_name     | WOE    | lower_bound | higher_bound | IV   |
|--------------|--------|-------------|--------------|------|
| EXT_SOURCE_3 | 1.046  | 0.001       | 0.228        | 0.33 |
| EXT_SOURCE_3 | 0.522  | 0.228       | 0.33         | 0.33 |
| EXT_SOURCE_3 | 0.182  | 0.33        | 0.408        | 0.33 |
| EXT_SOURCE_3 | -0.003 | 0.408       | 0.476        | 0.33 |
| EXT_SOURCE_3 | -0.311 | 0.476       | 0.564        | 0.33 |
| EXT_SOURCE_3 | -0.505 | 0.564       | 0.618        | 0.33 |
| EXT_SOURCE_3 | -0.666 | 0.618       | 0.694        | 0.33 |
| EXT_SOURCE_3 | -0.919 | 0.694       | 0.786        | 0.33 |
| EXT_SOURCE_3 | 0.156  | NULL        | NULL         | 0.33 |

The downside of less granular variable discretization is the inevitable reduction of information value of original variables.

However, compared with some other binning techniques widely utilized in the industry such as monotonic binning, this binning method has shown to retain much higher information value of the original variable even with reduced bin count. 

### Below is the EDA contingency table generated by R smbinning package for EXT_SOURCE_3 variable:

| Cutpoint             | CntRec | CntGood | CntBad | CntCumRec | CntCumGood | CntCumBad | PctRec | BadRate | Odds   | LnOdds  | WoE     | IV     |
|----------------------|--------|---------|--------|-----------|------------|-----------|--------|---------|--------|---------|---------|--------|
| <= 0.175605979469379 | 15394  | 3413    | 11981  | 15394     | 3413       | 11981     | 0.0501 | 0.7783  | 0.2849 | -1.2557 | 1.1768  | 0.1119 |
| <= 0.315472154925773 | 29572  | 4205    | 25367  | 44966     | 7618       | 37348     | 0.0962 | 0.8578  | 0.1658 | -1.7972 | 0.6353  | 0.0506 |
| <= 0.41184855592424  | 30781  | 2984    | 27797  | 75747     | 10602      | 65145     | 0.1001 | 0.9031  | 0.1073 | -2.2317 | 0.2008  | 0.0044 |
| <= 0.45789955120673  | 16008  | 1272    | 14736  | 91755     | 11874      | 79881     | 0.0521 | 0.9205  | 0.0863 | -2.4497 | -0.0172 | 0      |
| <= 0.502878277208218 | 17269  | 1238    | 16031  | 109024    | 13112      | 95912     | 0.0562 | 0.9283  | 0.0772 | -2.561  | -0.1285 | 0.0009 |
| <= 0.570916541772999 | 29890  | 1733    | 28157  | 138914    | 14845      | 124069    | 0.0972 | 0.942   | 0.0615 | -2.7879 | -0.3555 | 0.0106 |
| <= 0.636376171086044 | 31549  | 1512    | 30037  | 170463    | 16357      | 154106    | 0.1026 | 0.9521  | 0.0503 | -2.989  | -0.5565 | 0.0252 |
| <= 0.703203304904032 | 31367  | 1321    | 30046  | 201830    | 17678      | 184152    | 0.102  | 0.9579  | 0.044  | -3.1243 | -0.6919 | 0.0367 |
| <= 0.89600954949484  | 44716  | 1470    | 43246  | 246546    | 19148      | 227398    | 0.1454 | 0.9671  | 0.034  | -3.3816 | -0.9492 | 0.089  |
| Missing              | 60965  | 5677    | 55288  | 307511    | 24825      | 282686    | 0.1983 | 0.9069  | 0.1027 | -2.2761 | 0.1564  | 0.0052 |
| Total                | 307511 | 24825   | 282686 | NA        | NA         | NA        | 1      | 0.9193  | 0.0878 | -2.4325 | 0       | 0.3345 |

The graph below shows the information value of top 15 attributes' with 20-quantile equal bins (original_IV) vs. with optimized bins (updated_IV).

### Comparison of information value before & after binning optimization (Top 20 attributes)

![alt text](https://raw.githubusercontent.com/jtian24/EDA_Modeling_Spark/master/IV_comparison.png)


To confirm the correctness of the WOE binning and information value calculated, I also utilized R package smbinning to calculate the IV for top 10 numerical attributes and compare them with the result that I get:

![alt text](https://raw.githubusercontent.com/jtian24/EDA_Modeling_Spark/master/IV_comparison2.png)

As you can see, my result is mostly inline with the result coming from R package. It's worth noting that smbinning failed to calculate the WOE/IV for DAYS_LAST_PHONE_CHANGE attributes and get Inf result instead for unknown reason. Moreover, the average time it takes for the PySpark module to calcualte WOE/IV of each variable (with 307511 rows) is only 1.17s using 8 core CPUs, while for the R smbinning package it takes 54s! So the key advantage of this PySPark module is its high efficiency and streamlining of both data preprocessing, feature engineering and statistical modeling all within distributed computing framework. 

After all variables were optimally binned and encoded with WoE/mean value, the original dataset in PySpark dataframe was transformed with the a dictionary that maps original value to the WoE/mean value of its corresponding segment.It only took 71s to transform the original 300K X 123 dataset. 


(please reference transform_original_data function for modeling data WoE/Mean Value transformation using PySpark)

### Automated Logistic / Linear Regression Modeling with Elastic Net Regularization

The logistic regression building process is also automated for the sake of benchmarking and convenience. In reality, we need to double check if selected attributes are viable for a variety of other considerations, especially regulation considerations.

The logistic/linear regression model building is consisted of 2 options:

Option 1: Filter out variables with too low IV/R square, build the model using elastic net hyper-parameter tuning : Grid search of optimal penalty distribution (alpha) between L1 and L2 regularization and then at each alpha level (0.5 - 0.9), search for optimal lambda representing penalty strength to automatically select best set of variables in terms of cross validation performance. This option aims at creating optimal performance.

Option 2: Necessary for model scorecard building, build the model with variables pre-selected using IV/R square using IRLSM algorithm so as to get p-value for each variable. Drop any variable that has a p-value > 0.05 and then recursively refit the model until all variables of the logistic regression are statistically significant (p < 0.05)

(please reference optimal_glm_tuning and iterative_model_selection function for GLM grid search of hyper-parameters and model building)

Below is the coefficients table of the finalized logistic regression with option 2 using H2O GLM module:

| names              | coefficients | std_error   | z_value      | p_value     | standardized_coefficients |
|--------------------|--------------|-------------|--------------|-------------|---------------------------|
| Intercept          | -2.518710065 | 0.028041606 | -89.82046366 | 0           | -3.03634704               | 
| EXT_SOURCE_3       | 0.90264017   | 0.03612687  | 24.98528542  | 0           | 0.552413545               | 
| EXT_SOURCE_2       | 0.748842999  | 0.043063842 | 17.38913587  | 0           | 0.388219646               |
| EXT_SOURCE_1       | 0.633122487  | 0.041641299 | 15.20419637  | 0           | 0.363019165               |
| ORGANIZATION_TYPE  | 0.627250701  | 0.099469222 | 6.30597774   | 2.86E-10    | 0.158009536               | 
| OWN_CAR_AGE        | 0.628872652  | 0.104347376 | 6.026722217  | 1.67E-09    | 0.141618026               | 
| NAME_CONTRACT_TYPE | 0.916899548  | 0.223324089 | 4.1056903    | 4.03E-05    | 0.125941615               | 
| FLAG_DOCUMENT_3    | 0.310471353  | 0.14874064  | 2.087333711  | 0.03685798  | 0.056076214               | 
| CNT_CHILDREN       | -0.627442031 | 0.296022391 | -2.119576253 | 0.034041797 | -0.049089965              |

The logistic regression's performance on the validation dataset is shown below:

ModelMetricsBinomialGLM: glm  

| Performance Metrics         | Value    |
|-----------------------------|----------|
| MSE                         | 0.059    |
| RMSE                        | 0.242    |
| LogLoss                     | 0.221    |
| Null degrees of freedom     | 8172.000 |
| Residual degrees of freedom | 8164.000 |
| Null deviance               | 4025.725 |
| Residual deviance           | 3612.339 |
| AIC                         | 3630.339 |
| AUC                         | 0.742    |
| pr_auc                      | 0.186    |
| Gini                        | 0.485    |
  
### GBM modeling performance benchmark

In order to get a sense of how the above GLM model performs, GBM_model_eda module is implemented to create a hyper-parameter tuned Gradient Boosting Machine model in H2O to train the modeling dataset, and tested its performance on the testing dataset.

(please reference GBM_model_eda function for optimal GBM model building with hyper-parameter tuning)

To our surprise, the GBM model only performs marginally better (GBM AUC of 75.4% vs. GLM AUC of 74.2%) than the logistic regression model on the testing dataset, as shown below:

ModelMetricsBinomial: gbm

| Performance Metrics  | Values |
|----------------------|--------|
| MSE                  | 0.068  |
| RMSE                 | 0.261  |
| LogLoss              | 0.248  |
| Mean Per-Class Error | 0.312  |
| AUC                  | 0.754  |
| pr_auc               | 0.231  |
| Gini                 | 0.508  |

Meanwhile, the variable importance score of GBM model also reaffirms our GLM's variable selection for final model: 6 out of 8 variables selected by the final logistic regression model were among top 20 attributes by GBM variable importance, as shown below:

![alt text](https://raw.githubusercontent.com/jtian24/EDA_Modeling_Spark/master/gbm_top20_varimp.png)
