# High Performance EDA & GLM Modeling with Apache Spark and H2O

This PySpark/H2O based module (optimal_eda_glm.py) was created to automate the procedures of exploratory data analysis, numerical and categorical feature encoding and binning, optimal linear/logistic regression building with optional elasticnet hyper-parameter tuning, as well as GBM model building with hyperparameter tuning. 

With Apache Spark and H2O, this module is highly scalable to 'big data' and can process modeling dataset with millions of rows and thousands of variables with ease, as long as sufficient computing power (e.g., number of CPUs) is provided.

One of the key challenges (Fun Part) of this project is to research and develop 'optimal binning procedures' for numerical variables. The point of feature binning and optimization is to reduce the complexity of model scorecard widely used in financial services and credit industry while retaining as much predictive power of the original variables as possible.

Inspired by the method of conditional inference tree in r ctree package (https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf), I developed and tested a hypothesis testing based recursive partitioning approach for optimal binning for continuous numerical variables for both logistic and linear regression. The algorithm essentially build a classification tree based on single variable with respect to target variable. The root nodes of the tree will be the unique bins of each variable. Using the initial quantile based bins, the algorithm searches and identify optimal split point based on hypothesis testing statistics that signals maximum difference in proprtion (logistic regression) or mean value (linear regression). 

More specifically, Chi-square testing of independence is applied at each initial quantile bins for variables used by logistic regression. Welch T test of two independent sample mean with unknown variance is applied at each initial quantile bins for variables used by linear regression. Under both scenarios, split point with maximum testing statistics will be used for the recursive partitioning.

The classification tree continue to grow until two conditions are no longer met:

1. P-value of the testing statistics is higher than 0.05 (user defined).
2. The sample size of the bin is smaller than 5% of the total modeling sample size.

After the variable binning, Weight of Evidence (WOE) is calculated and used to replace the original feature value for their corresponding segments in logistic regression. Similarly, mean value is calculated and used to replace the original feature value for their corresponding segments in linear regression. The procedure will also calculate information value for each variable in logistic regression and R square statistic for each variable in linear regression. This way we are able to compare the univariate predictiveness of both numerical variables and categorical variables on the same basis, since for categorical variables WoE and IV is calculated directly based on predefined categories.

For demonstration, below is an example of using open source dataset from Kaggle Home Credit Default Risk competition to create a logistic regression model for default event prediction. Information value and WoE was calculated based on fixed width quantiles (e.g., 20-quantiles), below is the initial EDA result of top feature as measured by information value.

(Please reference the following functions: stack_columns_bucketization, init_woe_iv (logistic regression), init_r_square (linear regression) for initial quantile based binning and EDA procedures using PySpark/Pandas.)

### Preliminary top feature binning result (20-quantiles)


| var_name     | lower_bound | higher_bound | WOE   | overall_IV |
|--------------|-------------|--------------|-------|------------|
| EXT_SOURCE_3 | NULL        | NULL         | 1.169 | 0.42       |
| EXT_SOURCE_3 | 0.001       | 0.155        | 3.469 | 0.42       |
| EXT_SOURCE_3 | 0.155       | 0.228        | 2.272 | 0.42       |
| EXT_SOURCE_3 | 0.228       | 0.284        | 1.826 | 0.42       |
| EXT_SOURCE_3 | 0.284       | 0.33         | 1.544 | 0.42       |
| EXT_SOURCE_3 | 0.33        | 0.371        | 1.233 | 0.42       |
| EXT_SOURCE_3 | 0.371       | 0.408        | 1.168 | 0.42       |
| EXT_SOURCE_3 | 0.408       | 0.442        | 1.028 | 0.42       |
| EXT_SOURCE_3 | 0.442       | 0.476        | 0.97  | 0.42       |
| EXT_SOURCE_3 | 0.476       | 0.506        | 0.814 | 0.42       |
| EXT_SOURCE_3 | 0.506       | 0.535        | 0.697 | 0.42       |
| EXT_SOURCE_3 | 0.535       | 0.564        | 0.694 | 0.42       |
| EXT_SOURCE_3 | 0.564       | 0.592        | 0.593 | 0.42       |
| EXT_SOURCE_3 | 0.592       | 0.618        | 0.615 | 0.42       |
| EXT_SOURCE_3 | 0.618       | 0.643        | 0.531 | 0.42       |
| EXT_SOURCE_3 | 0.643       | 0.669        | 0.515 | 0.42       |
| EXT_SOURCE_3 | 0.669       | 0.694        | 0.496 | 0.42       |
| EXT_SOURCE_3 | 0.694       | 0.719        | 0.441 | 0.42       |
| EXT_SOURCE_3 | 0.719       | 0.749        | 0.396 | 0.42       |
| EXT_SOURCE_3 | 0.749       | 0.786        | 0.381 | 0.42       |


However, 20 bins is quite excessive from scorecard building perspective. Meanwhile, some neighboring bins do not have significantly different level of default risk (as measured by probability of default). By recursively paritioning the variable to identify the split point that maximizes the difference of default probability between neighboring segments as measured by Chi-square statistics, we are able to create simplified segments that retains as much predictiveness of the original attributes as possible, as shown below:  

(please reference update_iv_with_new_bin and recursive_var_bin function for recursive bin optimization for logistic regression, update_r_square_with_new_bin and LR_recursive_var_bin function for recursive bin optimization for linear regression)

### After Chi-squared testing based recursive partitioning of original bins (8 bins)

| var_name     | lower_bound | higher_bound | WOE    | overall_IV |
|--------------|-------------|--------------|--------|------------|
| EXT_SOURCE_3 | NULL        | NULL         | 0.156  | 0.330      |
| EXT_SOURCE_3 | 0.001       | 0.228        | 1.046  | 0.330      |
| EXT_SOURCE_3 | 0.228       | 0.330        | 0.522  | 0.330      |
| EXT_SOURCE_3 | 0.330       | 0.408        | 0.182  | 0.330      |
| EXT_SOURCE_3 | 0.408       | 0.476        | -0.003 | 0.330      |
| EXT_SOURCE_3 | 0.476       | 0.564        | -0.311 | 0.330      |
| EXT_SOURCE_3 | 0.564       | 0.618        | -0.505 | 0.330      |
| EXT_SOURCE_3 | 0.618       | 0.694        | -0.666 | 0.330      |
| EXT_SOURCE_3 | 0.694       | 0.786        | -0.919 | 0.330      |

The downside of less granular variable discretization is the inevitable reduction of information value of original variables.

However, compared with many binning techniques widely utilized in the industry (such as monotonic binning), this binning method has shown to retain much higher information value of the original variable.Below shows the information value of top 15 attributes' with 20-quantile equal bins vs. with optimized bins.

### Comparison of information value before & after binning optimization (Top 15 attributes)

![alt text](https://raw.githubusercontent.com/jtian24/EDA_Modeling_Spark/master/IV_comparison_plot.png)

After all variables were optimally binned and encoded with WoE, the original dataset in PySpark dataframe was transformed with the a dictionary that maps original value to the WoE value of its corresponding segment.

(please reference transform_original_data function for modeling data WoE/Mean Value transformation using PySpark)

### Automated Logistic / Linear Regression Modeling with Elastic Net Regularization

The logistic regression building process is also automated for the sake of benchmarking and convenience. In reality, we need to double check if selected attributes are viable for a variety of other considerations, especially regulation considerations.

The logistic/linear regression model building is consisted of 2 steps:

Step 1: Filter out variables with too low IV/R square, then build the model using elastic net hyper-parameter tuning : Grid search of optimal penalty distribution (alpha) between L1 and L2 regularization and then at each level of alpha search for optimal lambda representing penalty strength to automatically select best set of variables in terms of cross validation performance.

Step 2: Necessary for model scorecard building, re-fit the model with variables selected in Step 1 using IRLSM algorithm so as to get p-value for each variable. Drop any variable that has a p-value > 0.05 and then refit the model until all variables of the logistic regression are statistically significant.

(please reference optimal_glm_tuning and iterative_model_selection function for GLM grid search of hyper-parameters and model building)

Below is the coefficients table of the finalized logistic regression model:

| names              | coefficients | std_error   | z_value      | p_value     | standardized_coefficients | abs_std_coef |
|--------------------|--------------|-------------|--------------|-------------|---------------------------|--------------|
| Intercept          | -2.518710065 | 0.028041606 | -89.82046366 | 0           | -3.03634704               | 3.03634704   |
| EXT_SOURCE_3       | 0.90264017   | 0.03612687  | 24.98528542  | 0           | 0.552413545               | 0.552413545  |
| EXT_SOURCE_2       | 0.748842999  | 0.043063842 | 17.38913587  | 0           | 0.388219646               | 0.388219646  |
| EXT_SOURCE_1       | 0.633122487  | 0.041641299 | 15.20419637  | 0           | 0.363019165               | 0.363019165  |
| ORGANIZATION_TYPE  | 0.627250701  | 0.099469222 | 6.30597774   | 2.86E-10    | 0.158009536               | 0.158009536  |
| OWN_CAR_AGE        | 0.628872652  | 0.104347376 | 6.026722217  | 1.67E-09    | 0.141618026               | 0.141618026  |
| NAME_CONTRACT_TYPE | 0.916899548  | 0.223324089 | 4.1056903    | 4.03E-05    | 0.125941615               | 0.125941615  |
| FLAG_DOCUMENT_3    | 0.310471353  | 0.14874064  | 2.087333711  | 0.03685798  | 0.056076214               | 0.056076214  |
| CNT_CHILDREN       | -0.627442031 | 0.296022391 | -2.119576253 | 0.034041797 | -0.049089965              | 0.049089965  |

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

In order to get a sense of how the above GLM model performs, we also utilized GBM_model_eda module to create a hyper-parameter tuned Gradient Boosting Machine model to train the modeling dataset, and tested its performance on the testing dataset.

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
