# EDA_Modeling_Spark

This PySpark/H2O based module was created to automate the procedures of exploratory data analysis, numerical and categorical feature encoding and binning, as well as optimal linear/logistic regression building with optional elasticnet hyper-parameter tuning. With Apache Spark and H2O, this module is highly scalable to 'big data' and can process modeling dataset with hundreds of millions of rows and thousands of variables with ease, as long as sufficient computing power (e.g., number of CPUs) is provided.

One of the key challenges (Fun Part) of this project is to research and develop 'optimal binning procedures' for numerical variables. The point of feature binning and optimization is to reduce the complexity of model scorecard widely used in financial services and credit industry while retaining as much predictive power of individual variables being used. Instead of using raw feature value, Weight of Evidence (WOE) is calculated and used to replace the original feature value of their corresponding segments in logistic regression. Similarly, mean value is calculated and used to replace the original feature value of their corresponding segments in linear regression.

For demonstration, below is an example of using open source dataset from Kaggle Home Credit Default Risk competition to create a logistic regression model for default event prediction.

After initial EDA procedures, information value and WoE was calculated based on fixed quantile (e.g., 20-quantiles) with same length, below is the top feature as measured by information value.

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


However, 20 bins is quite excessive from scorecard building perspective. Meanwhile, some neighboring bins do not have significantly different level of default risk (as measured by probability of default). By recursively paritioning the variable to identify the split point that maximizes the difference of probability of default between population A and population B as measured by Chi-square statistics, we are able to create simplified segments that retains as much predictiveness of the original attributes as possible, as shown below:  

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

However, compared with many binning techniques widely utilized in the industry (such as monotonic binning), this binning method has retained much higher information value of the original variable. Below shows the information value of top 15 attributes' with 20-quantile equal bins vs. with optimized bins.

### Comparison of information value before & after binning optimization (Top 15 attributes)

![alt text](https://raw.githubusercontent.com/jtian24/EDA_Modeling_Spark/master/IV_comparison_plot.png)




