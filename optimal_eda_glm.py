
# coding: utf-8
##############################################################################################################################################
#Author: Jian Tian, last updated 12/2018
# This Python Module is designed to conduct exploratory data analysis, optimal feature encoding 
# and binning, as well as variable selection through both univariate analysis and GBM model building.
# 1. Feature Encoding Methods:
# Logistic Regression : WOE encoding, Linear Regression : Mean Encoding 
# 2. Univariate Analysis Metric for variable selection :
# Linear Regression: R square, Logistic Regression: Information Value (IV) 
# 3. Optimal Feature Binning Methods (reduce variable complexity for credit scorecard building):
# Linear Regression: Recursive partitioning (binary tree building) with Welch T Test of segment sample mean
# Logistic Regression: Recursive partitioning (binary tree building) with Chi-Square Test of Independence of sample proportion
# 4. Automatic GLM Variable Selection for both Linear / Logistic Regression with option to conduct elastic net hyper-parameter tuning:
#    Step1 : Identify optimal set of variables through GLM elastic net search (alpha = [0.01, 0.05, 0.1, 0.2], automatic beta search enabled)
#    Step2 : Constantly re-fit model with variables that has p value <.05 until all variables left are statistically significant
###############################################################################################################################################


import time 
import h2o
import pandas as pd
import numpy as np
import csv 
import pickle
import matplotlib
import matplotlib.pyplot as plt
import math
import unicodedata
from pyspark.sql.functions import when,udf
from pyspark.sql.functions import lit
from pyspark.sql.functions import rand,when
from pyspark.sql.types import BooleanType
from pyspark.sql.types import Row
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import Row
from pyspark.sql import SparkSession
from pyspark.storagelevel import *
from functools import reduce
from bisect import bisect
from pyspark.sql import functions as F
from pyspark.sql.dataframe import *
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from scipy.stats import chi2_contingency  
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
from pysparkling import *
from pyspark.sql.functions import col
from pysparkling import *
from pyspark.sql.functions import monotonically_increasing_id 
from scipy import stats
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

        
        

class optimal_eda_glm:
    
    def __init__(self):
        self.sc = sc
        self.hc = H2OContext.getOrCreate(sc) 
        self.sqlContext = SQLContext(sc)
        self.df = None
        self.target_var = None
        self.stringtype_list = []
        self.numericaltype_list = []
        self.constant_attributes = []
        self.all_null_attributes = []
        self.binned_categorical_attr = []
        self.samp_df = None
        self.categorical_attr = []
        self.numerical_attr = []
        self.attr_to_remove = []
        self.other_attributes = []
        self.df_total_cnt = 0
        self.df_samp_total_cnt = 0
        self.good_tot_cnt = 0
        self.bad_tot_cnt = 0
        self.dtype_dict = {}
        self.num_stacked_df = None
        self.cat_stacked_df = None
        self.num_var_quantile_dict = None
        self.cat_var_class_dict = None
        self.cat_var_dict = None
        self.init_woe_iv_info = None
        self.output_iv_df = pd.DataFrame()
        self.output_r_df = pd.DataFrame()
        self.h2o_df_samp = None
        self.segmented_var_dict = None
        self.binned_df_samp = None
        self.selected_numerical_attr = None
        self.selected_categorical_attr = None
        self.var_bin_dict = None
        self.var_woe_dict = None
        self.init_r_square_info = None
        self.final_var_list = None
        self.datatype_dict = None
        self.h2o_model_df = None
        self.h2o_train = None
        self.h2o_test = None
        self.h2o_valid = None
        self.optimal_glm = None
        self.optimal_coef_tbl = None
        
        
 
    def import_data(self, data_path='', infile_format='', sep=',', df=None,  target_var=None, create_sample=False, sample_cnt=50000,
                    increment_percent=0.05, linear_reg=False):

        if infile_format == 'csv' and data_path != '':
            self.df = self.sqlContext.read.format('com.databricks.spark.csv').option("sep",sep).options(header = 'true', inferschema = 'true').csv(data_path)
            
            self.df_total_cnt = self.df.count()
            self.target_var = target_var
            if create_sample == True:
                self.sample_ratio = sample_cnt/self.df_total_cnt
                self.df_samp = df.sample(withReplacement = False, fraction = self.sample_ratio, seed=1713)
                self.df_samp_total_cnt = self.df_samp.count()

            else:
                self.df_samp = self.df
                self.df_samp_total_cnt = self.df_samp.count()
                
            if linear_reg == False:
                self.good_tot_cnt = self.df_samp.filter(self.df_samp[self.target_var] > 0).count()
                self.bad_tot_cnt =  self.df_samp.filter(self.df_samp[self.target_var] == 0).count()
                
        else:
            self.df = df
            self.target_var = target_var
            self.df_total_cnt = df.count()
            if create_sample == True:
                self.sample_ratio = sample_cnt/self.df_total_cnt
                self.df_samp = df.sample(withReplacement = False, fraction = self.sample_ratio, seed=1713)
                self.df_samp_total_cnt = self.df_samp.count()
            else:
                self.df_samp = self.df
                self.df_samp_total_cnt = self.df_samp.count()
                
            if linear_reg == False:
                self.good_tot_cnt = self.df_samp.filter(self.df_samp[self.target_var] > 0).count()
                self.bad_tot_cnt =  self.df_samp.filter(self.df_samp[self.target_var] == 0).count()

             
             

        for f in self.df.schema.fields:
            self.dtype_dict[f.name] = str(f.dataType)
        
        #separate string type and numerical type
        for field in self.dtype_dict:
            if self.dtype_dict[field] in ['StringType']:
                    self.stringtype_list.append(field)

            else:
                    self.numericaltype_list.append(field)
                    
                    
        #initialize continuous variables and categorical variables initial binning statistics
        self.variable_type_identify(0.95)
        
        # cap categorical variables to maximum categories of 100, minimum percent represnented to be 95%
        self.cap_cat_var_class(0.95, 100)
                                 
        #calculate quantile bins for numerical attributes
        self.stack_columns_bucketization(increment_percent)
        

            
             
    
 #Identify percent of spark dataframe column that is NULL TYPE

    def null_percent(self):
        def check_null(s):
            if not s == None:
                return 0
            return 1

        is_null_udf = udf(check_null, IntegerType())       
        null_category_freq = self.df_samp.select(*(is_null_udf(col(c)).alias(c) for c in self.stringtype_list))
        null_category_freq_sum = null_category_freq.agg({column : "sum" for column in null_category_freq.columns}).collect()        
        null_category_freq_sum_df = self.sc.parallelize(null_category_freq_sum).toDF()        
        null_category_freq_sum_df = null_category_freq_sum_df.select([col(f).alias(f[4:-1]) for f in null_category_freq_sum_df.columns])       
        null_category_freq_sum_df = null_category_freq_sum_df.withColumn("row_cnt", lit(self.df_samp_total_cnt)) 
        null_category_freq_pct = null_category_freq_sum_df.select([(col(f)/col("row_cnt")).alias(f) for f in self.stringtype_list])
        null_category_freq_pct = null_category_freq_pct.toPandas().T
        null_category_freq_pct.rename(index=str, columns={0: 'null_pct'}, inplace = True)
        null_category_freq_pct['data_type'] = null_category_freq_pct.apply(lambda x: self.dtype_dict[x.name], axis=1)
        
        return null_category_freq_pct
    
    
#Identify percent of spark dataframe column that is numerical type (float, int, double, etc)
    
    def numerical_percent(self):
        
        def check_number(s): 
            if not s == None:
                try:
                    float(s)
                    return 1
                except ValueError:
                    pass
                try:
                    import unicodedata
                    unicodedata.numeric(s)
                    return 1
                except (TypeError, ValueError):
                    pass 
            return 0
 
        is_digit_udf = udf(check_number, IntegerType())
        num_category_freq = self.df_samp.select(*(is_digit_udf(col(c)).alias(c) for c in self.stringtype_list))
        num_category_freq_sum = num_category_freq.agg({column : "sum" for column in num_category_freq.columns}).collect()
        num_category_freq_sum_df = self.sc.parallelize(num_category_freq_sum).toDF()
        num_category_freq_sum_df = num_category_freq_sum_df.select(*(col(f).alias(f[4:-1]) for f in num_category_freq_sum_df.columns))
        num_category_freq_sum_df = num_category_freq_sum_df.withColumn("row_cnt", lit(self.df_samp_total_cnt))
        num_category_freq_pct = num_category_freq_sum_df.select(*((col(f)/col("row_cnt")).alias(f) for f in self.stringtype_list))
        num_category_freq_pct = num_category_freq_pct.toPandas().T 
        num_category_freq_pct.rename(index=str, columns={0: 'numerical_pct'}, inplace = True)
        num_category_freq_pct['data_type'] = num_category_freq_pct.apply(lambda x: self.dtype_dict[x.name], axis=1)
        
        return num_category_freq_pct



# Initialize summary statistics for later calculations
# steps to determine if a variable is continuous numerical, null/constant or categorical variable
# continous numerical: if variable is numerical => if this columns is double / integer type => continuous numerical attributes
# NULL/Constanat attributes: if all elements of this column are constant or None
# categorical attributes: if more than 95% of elements are string version of numerical value, then force conversion to numerical   variable, otherwise it is categorical attributes
        
    def variable_type_identify(self, true_num_pct_threshold):

        str_var_null_freq = self.null_percent()
        str_var_num_freq =  self.numerical_percent()
        str_var_null_num_var_freq = pd.merge(str_var_null_freq, str_var_num_freq, left_index=True, right_index = True)
        str_var_null_num_var_freq = str_var_null_num_var_freq.drop(['data_type_y'], axis = 1)
        str_var_null_num_var_freq = str_var_null_num_var_freq.rename({'data_type_x':'data_type'}, axis = 1)
        str_var_null_num_var_freq['total_percent'] = 1.0
        str_var_null_num_var_freq['true_num_pct'] = str_var_null_num_var_freq.apply(lambda row: row['numerical_pct']/(row['total_percent'] - row['null_pct']) if row['total_percent'] - row['null_pct'] > 0 else row['numerical_pct'], axis = 1)
        numerical_attributes = [str(x) for x in str_var_null_num_var_freq[str_var_null_num_var_freq['true_num_pct'] >= true_num_pct_threshold].index]
        self.all_null_attributes = [str(x) for x in str_var_null_num_var_freq[str_var_null_num_var_freq['null_pct'] == 1].index]
        string_attributes = []

        
        for x in self.stringtype_list: 
            if x not in numerical_attributes + self.all_null_attributes: 
                unique_char_value_set = self.df_samp.select(x).distinct().rdd.map(lambda r: r[0]).collect()
                if unique_char_value_set == 1:
                    self.constant_attributes.append(x)
                else:
                    string_attributes.append(x)
                    if None in unique_char_value_set:
                        self.df_samp = self.df_samp.fillna('NULL', subset=[x])
 
        for var in self.numerical_attr:
            self.df= self.df.withColumn(var, self.df[var].cast(DoubleType()))
            self.df_samp = self.df_samp.withColumn(var, self.df_samp[var].cast(DoubleType()))
        
        self.categorical_attr = string_attributes
        self.numerical_attr = numerical_attributes + self.numericaltype_list
        self.h2o_df = self.hc.as_h2o_frame(self.df)
        self.h2o_df_samp = self.hc.as_h2o_frame(self.df_samp)
        self.datatype_dict = {'numerical_attributes:': self.numerical_attr, 'categorical_attributes:': self.categorical_attr, 'constant attributes :': self.constant_attributes}
            
         

            
        
            

    
        
    
#identify percentile/increment cut point value of each continuous variable for initial binning
    
    def stack_columns_bucketization(self,  increment, df=None, numerical_attr_names=None, categorical_attr_names=None,  target_var=None):
        if df == None:
            df = self.df_samp
            
        if numerical_attr_names == None:
            numerical_attr_names = self.numerical_attr

            
        if categorical_attr_names == None:
            categorical_attr_names = self.categorical_attr
            
        if target_var == None:
            target_var = self.target_var 
            
        def is_number(s):
            if s == None:
                return False
            try:
                float(s) or int(s)
                return True
            except ValueError:
                return False
  
        all_num_h2o_df = self.h2o_df_samp[:, numerical_attr_names]
        df_quantile = all_num_h2o_df[:,numerical_attr_names[0]].quantile(prob = np.arange(0,1,increment).tolist()).drop('Probs').set_names([numerical_attr_names[0]])
        
        for attr in numerical_attr_names[1:]:
            another_quantile = all_num_h2o_df[:,attr].quantile(prob = np.arange(0,1,increment).tolist()).drop('Probs').set_names([attr])
            df_quantile = df_quantile.cbind(another_quantile)
            
        pd_df_quantile = df_quantile.as_data_frame()
        
        num_attr_quantile_dict = {}
        for attr in pd_df_quantile.columns.values:
            quantile_list = sorted(list(set([float(x) for x in pd_df_quantile[attr].values.tolist() if not math.isnan(x)])))
            if not len(quantile_list) == 0:
                num_attr_quantile_dict.setdefault(attr)
                num_attr_quantile_dict[attr] = quantile_list
        
                
        #remove 'all nan' attribute
        updated_attr_quantile = {}
        for k, v in num_attr_quantile_dict.items():
            if k != 'Probs':
                if not math.isnan(list(set(v))[0]):
                    updated_attr_quantile[k] = [x for x in v]
                    
        self.num_var_quantile_dict = updated_attr_quantile
        
# cap the total number of distinct categories and the minimum  % represented by each categorical variable        
    def cap_cat_var_class(self, max_class_percent, max_nbr_of_class):

        df = self.df_samp
        target_var = self.target_var
        categorical_attr_names = self.categorical_attr            
        h2o_df =  self.hc.as_h2o_frame(df.select(categorical_attr_names))
        
        catfeat_class_dict = {}
        
        for catfeat in categorical_attr_names:
            #convert categorical (string based variables) to factor data type
            h2o_df[catfeat] = h2o_df[catfeat].asfactor()
            #calculate frequency table of each class for the categorical variable 
            cat_attr_freq = h2o_df.group_by(catfeat).count().get_frame().as_data_frame()
            #sort columes of the categorical class frequency from highest to lowest 
            catfeat_class_dict.setdefault(catfeat)
            catfeat_class_dict[catfeat] = []
            total_cnt = cat_attr_freq['nrow'].sum()
            cat_attr_freq['percent'] = cat_attr_freq.apply(lambda row: row['nrow']/float(total_cnt), axis = 1)
            cat_attr_freq = cat_attr_freq.sort_values('nrow', ascending = False).reset_index(drop = True)
            
            max_index = 0
            total_pct = 0
        
            while total_pct < max_class_percent and max_index < max_nbr_of_class:
                total_pct = total_pct + cat_attr_freq.loc[max_index, 'percent']
                catfeat_class_dict[catfeat].append(cat_attr_freq.loc[max_index, catfeat])
                max_index = max_index + 1
  
        self.cat_var_class_dict = catfeat_class_dict
        
        #replace categorical variables' element that ranks lower than max_nbr_of_class or contribute less than 1 - max_class_percent to total population
        def simplify_categorical_udf(catfeat_class_dict):
            return udf(lambda var, var_name: var if var in catfeat_class_dict[var_name] else 'other', StringType())
        
        for cat_attr in self.cat_var_class_dict:  
            capped_cat_var = cat_attr + '_capped' 
            self.df_samp = self.df_samp.withColumn(capped_cat_var, simplify_categorical_udf(self.cat_var_class_dict)(col(cat_attr).alias('var'), lit(cat_attr).alias('var_name')))
            self.binned_categorical_attr.append(capped_cat_var)

        
        

       
            
#initial binning for WOE and Information Value calculation for each numerical attributes for logistic regression
    
    def init_woe_iv(self, numerical_attr=None, categorical_attr=None):
        #convert df to column stacked long dataset to apply function on them  
        if numerical_attr == None:
            numerical_attr = self.num_var_quantile_dict.keys()
            
        if categorical_attr == None:
            categorical_attr = self.categorical_attr
            
        self.selected_numerical_attr = numerical_attr
        self.selected_categorical_attr = categorical_attr
            
        #define continuous and categorical dataframe    
        continuous_df_col_list = []
        cat_df_col_list = [] 
        df_col_list = []
        
        def bucketize(var,  var_name, quantile_dict):
            if var == None:
                return 'NULL'
            quantiles = quantile_dict[var_name]
            
            if len(quantiles) > 2:
                    if bisect(quantiles, var) <= len(quantiles) - 1:
                        higher_bound = quantiles[bisect(quantiles, var)]
                        lower_bound = quantiles[bisect(quantiles, var) - 1]
                    else:
                   
                        higher_bound = quantiles[len(quantiles) - 1]
                        lower_bound = quantiles[len(quantiles) - 2]
            else:
                    if bisect(quantiles, var) > 0:
                        higher_bound = quantiles[bisect(quantiles, var)-1]
                        lower_bound = quantiles[bisect(quantiles, var)-1]
                    else:
                        higher_bound = quantiles[0]
                        lower_bound = quantiles[0]
                       

                    
            return str(lower_bound) + ' - ' + str(higher_bound)

        def find_segment_udf(quantile_dict):
            return udf(lambda var, var_name: bucketize(var, var_name, quantile_dict), StringType())
        
        
        def simplify_categorical_udf(catfeat_class_dict):
            return udf(lambda var, var_name: var if var in catfeat_class_dict[var_name] else 'other', StringType())
        
        segmented_var_dict = {} 
        segmented_var_list = []        
        self.df_samp = self.df_samp.withColumn("id", monotonically_increasing_id())
         
        if len(numerical_attr) > 0:
            for attr in numerical_attr:
                if attr not in ['lexid', self.target_var, 'Prob']:
                    segmented_var_dict.setdefault(attr)
                    segmented_var_dict[attr] = self.df_samp.select([find_segment_udf(self.num_var_quantile_dict)(col(attr).alias('var'),lit(attr).alias('var_name')).alias('segment'), col(self.target_var).alias('target_var')]).groupBy('segment').agg(F.sum('target_var').alias('nonzero_cnt'), F.count('*').alias('total_count')).sort(['segment']).toPandas()
                    segmented_var_dict[attr]['var_name'] = attr
                
        cat_start_time = time.time()        

        if len(categorical_attr) > 0:
            for attr in categorical_attr:
                if attr not in ['lexid', self.target_var]:
                    segmented_var_dict.setdefault(attr)
                    segmented_var_dict[attr] = self.df_samp.select([udf(lambda var: str(var) + ' - ' + str(var), StringType())(col(attr).alias('var')).alias('segment'), col(self.target_var).alias('target_var')]).groupBy('segment').agg(F.sum('target_var').alias('nonzero_cnt'), F.count('*').alias('total_count')).sort(['segment']).toPandas()
                    segmented_var_dict[attr]['var_name'] = attr

        
        freq_df_list = [] 
        for df_key in segmented_var_dict.keys():
            freq_df_list.append(segmented_var_dict[df_key])
            
        freq_tbl = pd.concat(freq_df_list, ignore_index=True)
        freq_tbl['zero_cnt'] = freq_tbl.apply(lambda row: row['total_count'] - row['nonzero_cnt'], axis = 1)
        freq_tbl['good_total'] = self.good_tot_cnt
        freq_tbl['bad_total'] = self.bad_tot_cnt
        freq_tbl['good_pct'] = freq_tbl.apply(lambda row: float(row['nonzero_cnt'] + 0.5)/(row['good_total'] + 0.5), axis = 1)
        freq_tbl['bad_pct'] = freq_tbl.apply(lambda row: float(row['zero_cnt'] + 0.5)/(row['bad_total'] + 0.5), axis = 1)
        freq_tbl['WOE'] = freq_tbl.apply(lambda row: float(row['good_pct'])/row['bad_pct'], axis = 1)
        freq_tbl['IV'] = freq_tbl.apply(lambda row: (row['good_pct'] - row['bad_pct'])*row['WOE'], axis = 1)
        agg_freq_tbl = freq_tbl.groupby(['var_name']).agg({'IV': 'sum'}).reset_index().rename(index=str, columns={"IV": "IV_sum"}).sort_values('IV_sum', ascending = False)
        freq_tbl = freq_tbl.join(agg_freq_tbl.set_index('var_name'), on = 'var_name')
        
        self.init_woe_iv_info = freq_tbl

        return freq_tbl
 
#Hypothesis testing (Chi-Square Test of independence) based recursive partitioning of each sorted numerical varaible into segments that has significant difference in proportion of sample being positive

    def recursive_var_bin(self, pd_num_freq_tbl1, total_population_cnt, p_threshold,sub_population_pct, cut_point_list, chi_square_list):
        from scipy.stats import chi2_contingency
        if pd_num_freq_tbl1.empty == True:
            return
        if pd_num_freq_tbl1.total_count.sum()/float(total_population_cnt) <= sub_population_pct:
            return
        #stopping criteria: 1. if sub population cnt <= total_population_cnt 2. if chi square test p value > p_threshold
        #step 1, sort the dataframe by lower bound
        pd_num_freq_tbl1 = pd_num_freq_tbl1.sort_values(by = 'lower_bound', axis = 0)
        #step 2, calculate <= lower bound and >= lower bound population count, nonzero_cnt, zero_cnt
        pd_num_freq_tbl1.loc[:, 'p_value'] = 0
        pd_num_freq_tbl1.loc[:, 'dof'] = 0
        pd_num_freq_tbl1.loc[:, 'chi2'] = 0
        for index, row in pd_num_freq_tbl1.iterrows():
            lb = row['lower_bound']
            left_nonzero = int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] <= lb, 'nonzero_cnt'].sum()) if int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] <= lb, 'nonzero_cnt'].sum()) > 1 else 1
            left_zero =  int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] <= lb, 'zero_cnt'].sum()) if int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] <= lb, 'zero_cnt'].sum()) > 1 else 1
            right_nonzero = int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] >= lb, 'nonzero_cnt'].sum()) if int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] >= lb, 'nonzero_cnt'].sum()) > 1 else 1
            right_zero = int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] >= lb, 'zero_cnt'].sum()) if int(pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] >= lb, 'zero_cnt'].sum()) > 1 else 1
            obs = np.array([[left_nonzero, left_zero],
                    [right_nonzero, right_zero]])
            chi2, p, dof, expctd = chi2_contingency(obs) 
            pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'p_value'] = p
            pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'dof'] = dof
            pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'chi2'] = chi2
        split_val = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['chi2'].idxmax(), 'lower_bound']
        split_p = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['chi2'].idxmax(), 'p_value']
        sub_df1 = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['lower_bound'] <= split_val]
        sub_df2 = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['lower_bound'] > split_val]
        max_chi_square = pd_num_freq_tbl1['chi2'].max()

        if split_p > p_threshold:
            return
        if not split_val in cut_point_list:
            cut_point_list.append(split_val)
            chi_square_list.append(max_chi_square)
            self.recursive_var_bin(sub_df1, total_population_cnt, p_threshold, sub_population_pct, cut_point_list, chi_square_list)
            self.recursive_var_bin(sub_df2, total_population_cnt, p_threshold, sub_population_pct, cut_point_list, chi_square_list)
        else:
            return    
    
    
# update numerical variable's binning to reduce the number of total segments of each numercial variable while retaining as much information value of each numerical variable as possible for later logistic regression
    
    def update_iv_with_new_bin(self, p_threshold, sub_population_pct, selected_attributes=[]):
        
        def bucketize(var,  var_name, quantile_dict):
            if var == None:
                return 'NULL - NULL'
            
            if var == -999:
                return '-999 - -999'
            quantiles = quantile_dict[var_name]['sorted_bin_seq']
            
            if len(quantiles) > 2:
                    if bisect(quantiles, var) <= len(quantiles) - 1:
                        higher_bound = quantiles[bisect(quantiles, var)]
                        lower_bound = quantiles[bisect(quantiles, var) - 1]
                    else:
                   
                        higher_bound = quantiles[len(quantiles) - 1]
                        lower_bound = quantiles[len(quantiles) - 2]
            else:
                    if bisect(quantiles, var) > 0:
                        higher_bound = quantiles[bisect(quantiles, var)-1]
                        lower_bound = quantiles[bisect(quantiles, var)-1]
                    else:
                        higher_bound = quantiles[0]
                        lower_bound = quantiles[0]
                       

                    
            return str(lower_bound) + ' - ' + str(higher_bound)
        
        var_bin_dict = {}
        
        if len(selected_attributes) >= 1 and selected_attributes != None:
            pd_num_freq_tbl = self.init_woe_iv_info.loc[self.init_woe_iv_info['var_name'].isin(selected_attributes)]
        else:
            pd_num_freq_tbl = self.init_woe_iv_info
            
        #this is numerical attributes
        pd_num_freq_tbl0 = pd_num_freq_tbl.loc[pd_num_freq_tbl['var_name'].isin(self.num_var_quantile_dict.keys())]
        #this is categorical attributes
        pd_num_freq_tbl1 = pd_num_freq_tbl.loc[pd_num_freq_tbl['var_name'].isin(self.num_var_quantile_dict.keys()) == False]
        number_of_cat_var = pd_num_freq_tbl['var_name'].isin(self.num_var_quantile_dict.keys()) == False
        pd_num_freq_tbl0.replace(to_replace='NULL', value= '-999 - -999', inplace = True)
        
        for var in [str(x) for x in list(set(pd_num_freq_tbl0.var_name))]:
            if var in self.num_var_quantile_dict.keys():
                var_bin_dict.setdefault(var)
                var_bin_dict[var] = {}
                cut_point_list = []
                chi_square_list = []
                df = pd_num_freq_tbl0.loc[pd_num_freq_tbl0['var_name'] == var, :]
                if df.shape[0] == 1:
                    df.loc[:, 'lower_bound'] = df.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[0]))
                    cut_point_list.append(int(df.loc[:, 'lower_bound']))
                    chi_square_list.append(0)
                    var_bin_dict[var]['bin_seq'] = cut_point_list
                    var_bin_dict[var]['sorted_bin_seq'] = cut_point_list
                    var_bin_dict[var]['chi_square_seq'] = chi_square_list
                else:
                    df.loc[:, 'lower_bound'] = df.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[0]))
                    df.loc[:, 'higher_bound'] = df.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[1]))
                    total_population_cnt = df.total_count.sum()
                    self.recursive_var_bin(df, total_population_cnt, p_threshold,sub_population_pct, cut_point_list, chi_square_list)
                    true_min_lower_bound = df.loc[:, 'lower_bound'].min()
                    true_max_higher_bound = df.loc[:, 'higher_bound'].max()
                    var_bin_dict[var]['bin_seq'] = cut_point_list
                    cut_point_list = cut_point_list + [true_min_lower_bound,true_max_higher_bound]
                    var_bin_dict[var]['sorted_bin_seq'] = sorted(list(set(cut_point_list)))
                    var_bin_dict[var]['chi_square_seq'] = chi_square_list
                    
        for var in var_bin_dict:
            var_bin_dict[var]['chi_square_dict'] = dict(zip([str(x) for x in var_bin_dict[var]['bin_seq']],  var_bin_dict[var]['chi_square_seq']))
            var_bin_dict[var]['binning_sequence'] = dict(zip([str(x) for x in var_bin_dict[var]['bin_seq']], range(len(var_bin_dict[var]['chi_square_seq']))))
    
        self.var_bin_dict = var_bin_dict
        pd_num_freq_tbl0.loc[:, 'lower_bound'] = pd_num_freq_tbl0.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[0]))
        pd_num_freq_tbl0.loc[:, 'higher_bound'] = pd_num_freq_tbl0.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[1]))
        pd_num_freq_tbl0['new_bin'] = pd_num_freq_tbl0.apply(lambda row: bucketize(row['lower_bound'], row['var_name'], var_bin_dict), axis = 1)            
        if number_of_cat_var.sum() > 0:
            pd_num_freq_tbl1['lower_bound'] = pd_num_freq_tbl1['segment']
            pd_num_freq_tbl1['higher_bound'] = pd_num_freq_tbl1['segment']
            pd_num_freq_tbl1['new_bin'] = pd_num_freq_tbl1.apply(lambda row: row['lower_bound'] + ' - ' + row['higher_bound'], axis = 1)
            pd_freq_tbl = pd.concat([pd_num_freq_tbl0, pd_num_freq_tbl1])
        else:
            pd_freq_tbl = pd_num_freq_tbl0
        new_pd_num_freq = pd_freq_tbl.groupby(['var_name','new_bin']).agg({'nonzero_cnt': np.sum, 'zero_cnt': np.sum })
        sub_var_df_list = []
        for var_name in new_pd_num_freq.index.get_level_values(0).unique():
                sub_var_df = new_pd_num_freq.loc[var_name,:]
                sub_var_df.loc[:, 'new_bin'] = sub_var_df.index
                sub_var_df.loc[:, 'lower_bound'] = sub_var_df.apply(lambda row: row['new_bin'].split(' - ')[0], axis = 1)
                sub_var_df.loc[:, 'higher_bound'] = sub_var_df.apply(lambda row: row['new_bin'].split(' - ')[1], axis = 1)
                sub_var_df.loc[:, 'var_name'] = str(var_name)
                sub_var_df.loc[:, 'good_total'] = self.good_tot_cnt
                sub_var_df.loc[:, 'bad_total'] = self.bad_tot_cnt
                sub_var_df.loc[:, 'good_pct'] = sub_var_df.apply(lambda row: (row['nonzero_cnt'] + 0.5)/(float(row['good_total']) + 0.5), axis = 1)
                sub_var_df.loc[:, 'bad_pct'] = sub_var_df.apply(lambda row: (row['zero_cnt'] + 0.5)/(float(row['bad_total']) + 0.5), axis = 1)
                sub_var_df.loc[:, 'index'] = sub_var_df.apply(lambda row: int(100*row['good_pct']/row['bad_pct']) , axis = 1)
                sub_var_df.loc[:, 'WOE'] = sub_var_df.apply(lambda row: math.log(row['good_pct']/row['bad_pct']), axis = 1)
                sub_var_df.loc[:, 'IV_bucket'] = sub_var_df.apply(lambda row: float(row['WOE']*(row['good_pct'] - row['bad_pct'])), axis = 1)
                sub_var_df_list.append(sub_var_df)
        new_pd_num_freq1 = pd.concat(sub_var_df_list)
        new_pd_num_iv = new_pd_num_freq1.groupby('var_name').agg({'IV_bucket': np.sum})
        new_pd_num_iv = new_pd_num_iv.rename(index=str, columns={"IV_bucket": "IV"})
        new_pd_num_freq2 = new_pd_num_freq1.set_index('var_name').join(new_pd_num_iv, how = 'left')
        new_pd_num_freq2.loc[:, 'var_name'] = new_pd_num_freq2.index
        new_pd_num_freq2.loc[:, 'num_lower_bound'] = new_pd_num_freq2.apply(lambda row: float(row['lower_bound']) if row['var_name'] in self.num_var_quantile_dict.keys() else 0, axis = 1)
        new_pd_num_freq2.loc[:, 'var_type'] = new_pd_num_freq2.apply(lambda row: 'numerical' if row['var_name'] in self.num_var_quantile_dict.keys() else 'categorical', axis = 1)
        new_pd_num_freq2 = new_pd_num_freq2.sort_values(by = ['IV', 'var_name', 'num_lower_bound'],ascending = [0,1, 1])
        output_iv_df = new_pd_num_freq2[['var_type','lower_bound', 'higher_bound', 'good_pct', 'bad_pct', 'nonzero_cnt', 'zero_cnt', 'good_total', 'bad_total', 'WOE', 'IV_bucket','index', 'IV']]
        
        def find_chi_square(row, chi_square_dict):
            if row.name in chi_square_dict:
                chi_square = chi_square_dict[row.name]['chi_square_dict'][row['lower_bound']] if row['lower_bound'] in chi_square_dict[row.name]['chi_square_dict'] else 0
            else:
                chi_square = 0
            return chi_square
        
        def find_binning_seq(row, chi_square_dict):
            if row.name in chi_square_dict:
                binning_seq = chi_square_dict[row.name]['binning_sequence'][row['lower_bound']] if row['lower_bound'] in chi_square_dict[row.name]['chi_square_dict'] else 999
            else:
                binning_seq = 999
                
            return binning_seq
        
        
        output_iv_df.loc[:, 'chi_square'] = output_iv_df.apply(find_chi_square, args = (var_bin_dict,), axis = 1)
        output_iv_df.loc[:, 'binning_seq'] = output_iv_df.apply(find_binning_seq, args = (var_bin_dict,), axis = 1)
        
        self.output_iv_df = output_iv_df
        
            
        self.final_var_list = list(output_iv_df.index.unique())
        output_iv_df['segment'] = output_iv_df.apply(lambda row: str(row['lower_bound']) + " - " + str(row['higher_bound']), axis = 1)        
        
        self.univariate_varimp = output_iv_df.loc[:, ['var_name', 'IV']].drop_duplicates().sort_values(by = ['IV'], ascending = False)
        
        var_woe_dict = {}
        for variable in self.final_var_list:
            var_woe_dict.setdefault(variable)
            var_woe_dict[variable] = output_iv_df.loc[output_iv_df.index == variable,['segment','WOE']].set_index('segment').to_dict()['WOE']
            
        self.var_encoding_dict = var_woe_dict    
        
        return output_iv_df
    
    
 #initial binning for numerical attributes' mean encoding and varaible R square calculation based on predefined increment value
    
    def init_r_square(self, numerical_attr=None, categorical_attr=None, excluded_var=None):
            #convert df to column stacked long dataset to apply function on them  
            if numerical_attr == None:
                numerical_attr = self.num_var_quantile_dict.keys()
            
            if categorical_attr == None:
                categorical_attr = self.categorical_attr
            
            self.selected_numerical_attr = [var for var in numerical_attr if var in self.num_var_quantile_dict]
            self.selected_categorical_attr = [var for var in categorical_attr if var in self.categorical_attr]
            
            #define continuous and categorical dataframe    
            continuous_df_col_list = []
            cat_df_col_list = []
        
            #population_cnt = h2o_df.shape[0]
            df_col_list = []
        
        
            def bucketize(var,  var_name, quantile_dict):
                if var == None:
                    return 'NULL - NULL'
                quantiles = quantile_dict[var_name]
            
                if len(quantiles) > 2:
                    if bisect(quantiles, var) <= len(quantiles) - 1:
                        higher_bound = quantiles[bisect(quantiles, var)]
                        lower_bound = quantiles[bisect(quantiles, var) - 1]
                    else:
                   
                        higher_bound = quantiles[len(quantiles) - 1]
                        lower_bound = quantiles[len(quantiles) - 2]
                else:
                    if bisect(quantiles, var) > 0:
                        higher_bound = quantiles[bisect(quantiles, var)-1]
                        lower_bound = quantiles[bisect(quantiles, var)-1]
                    else:
                        higher_bound = quantiles[0]
                        lower_bound = quantiles[0]
                                     
                return str(lower_bound) + ' - ' + str(higher_bound)
                    
            def find_segment_udf(quantile_dict):
                return udf(lambda var, var_name: bucketize(var, var_name, quantile_dict), StringType())
        
        
            def simplify_categorical_udf(catfeat_class_dict):
                return udf(lambda var, var_name: var if var in catfeat_class_dict[var_name] else 'other', StringType())
            
            segmented_var_dict = {}
        
            self.df_samp = self.df_samp.withColumn("id", monotonically_increasing_id())

            target_mean = self.df_samp.agg({self.target_var: "avg"}).collect()[0][0]
            
            if len(numerical_attr) > 0:
                for attr in numerical_attr:
                    if attr not in excluded_var + [self.target_var]:
                        segmented_var_dict.setdefault(attr)
                        segmented_var_dict[attr] = self.df_samp.select([find_segment_udf(self.num_var_quantile_dict)(col(attr).alias('var'),lit(attr).alias('var_name')).alias('segment'), udf(lambda var: (var - target_mean)**2 if not var == None else 0, DoubleType())(col(self.target_var)).alias('mean_diff_square'), col(attr).alias('var'), col(self.target_var).alias('target_var')]).groupBy('segment').agg(F.sum('target_var').alias('target_var_sum'), F.count('*').alias('total_count'), F.avg('target_var').alias('target_mean'),  F.var_samp('target_var').alias('sample_variance'), F.sum('mean_diff_square').alias('segment_total_var')).sort(['segment']).toPandas()
                        segmented_var_dict[attr]['segment_explained_var'] = segmented_var_dict[attr]['total_count']*(segmented_var_dict[attr]['target_mean'] - target_mean)**2
                        segmented_var_dict[attr]['r_square'] = segmented_var_dict[attr].apply(lambda row: float(row['segment_explained_var'])/float(row['segment_total_var']) if not float(row['segment_total_var']) == 0 else 0, axis = 1)
                        segmented_var_dict[attr]['var_name'] = attr
                        total_r_square = float(segmented_var_dict[attr]['segment_explained_var'].sum())/float(segmented_var_dict[attr]['segment_total_var'].sum())
                        segmented_var_dict[attr]['tot_r_square'] = total_r_square

            if len(categorical_attr) > 0:
                for attr in categorical_attr:
                    if attr not in excluded_var + [self.target_var]:
                        segmented_var_dict.setdefault(attr)
                        segmented_var_dict[attr] = self.df_samp.select([udf(lambda var: var + ' - '+ var, StringType())(col(attr)).alias('segment'), udf(lambda var: (var - target_mean)**2  if not var == None else 0, DoubleType())(col(self.target_var)).alias('mean_diff_square'), col(attr).alias('var'), col(self.target_var).alias('target_var')]).groupBy('segment').agg(F.sum('target_var').alias('target_var_sum'), F.count('*').alias('total_count'), F.avg('target_var').alias('target_mean'), F.var_samp('target_var').alias('sample_variance'), F.sum('mean_diff_square').alias('segment_total_var')).sort(['segment']).toPandas()
                        segmented_var_dict[attr]['segment_explained_var'] = segmented_var_dict[attr]['total_count']*(segmented_var_dict[attr]['target_mean'] - target_mean)**2
                        segmented_var_dict[attr]['r_square'] = segmented_var_dict[attr].apply(lambda row: float(row['segment_explained_var'])/float(row['segment_total_var']) if not float(row['segment_total_var']) == 0 else 0, axis = 1)
                        segmented_var_dict[attr]['var_name'] = attr
                        total_r_square = float(segmented_var_dict[attr]['segment_explained_var'].sum())/float(segmented_var_dict[attr]['segment_total_var'].sum())
                        segmented_var_dict[attr]['tot_r_square'] = total_r_square

            freq_df_list = []
        
            for df_key in segmented_var_dict.keys():
                 freq_df_list.append(segmented_var_dict[df_key])
            
            freq_tbl = pd.concat(freq_df_list, ignore_index=True)
            
            self.init_r_square_info = freq_tbl

            return freq_tbl


  
# Hypothesis testing (Welch T test of sample mean) based recursive partitioning of continuous numerical variable into optimal bins 
    
    def LR_recursive_var_bin(self, pd_num_freq_tbl1, target_var_array, p_threshold, sub_population_pct, cut_point_list, t_value_list, p_value_list):
        import math
        from scipy import stats
        if pd_num_freq_tbl1.empty == True:
            return
        if pd_num_freq_tbl1.total_count.sum()/float(self.df_samp_total_cnt) <= sub_population_pct:
            return
        #stopping criteria: 1. if sub population cnt <= total_population_cnt 2. if chi square test p value > p_threshold
        #step 1, sort the dataframe by lower bound
        pd_num_freq_tbl1 = pd_num_freq_tbl1.sort_values(by = 'lower_bound', axis = 0)
        
        #step 2, calculate <= lower bound and >= lower bound population count, nonzero_cnt, zero_cnt
        pd_num_freq_tbl1.loc[:, 'p_value'] = 0
        pd_num_freq_tbl1.loc[:, 'abs_t_value'] = 0
        for index, row in pd_num_freq_tbl1.iterrows():
            lb = row['lower_bound']
            left_target_var = target_var_array[target_var_array[:,0] <= lb][:, 1]
            right_target_var = target_var_array[target_var_array[:,0] > lb][:, 1] 
            left_total_count = left_target_var.shape[0]
            right_total_count = right_target_var.shape[0]

            if left_total_count >= int(self.df_samp_total_cnt*sub_population_pct) and right_total_count >= int(self.df_samp_total_cnt*sub_population_pct):
                #compute pooled variance and mean for those below lb
                t_statistics, p_value = stats.ttest_ind(left_target_var, right_target_var, equal_var = False)
                pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'p_value'] = p_value
                pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'abs_t_value'] = t_statistics if t_statistics > 0 else -t_statistics
            else:
                pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'p_value'] = 1
                pd_num_freq_tbl1.loc[pd_num_freq_tbl1.loc[:, 'lower_bound'] == lb, 'abs_t_value'] = 0
                continue
        
        pd_num_freq_tbl1 = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['p_value'] < 1, :]
        if pd_num_freq_tbl1.shape[0] > 2:
            split_val = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['abs_t_value'].idxmax(), 'lower_bound']
            split_p = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['abs_t_value'].idxmax(), 'p_value']
            left_target_var_array = target_var_array[target_var_array[:,0] <= split_val]
            right_target_var_array = target_var_array[target_var_array[:,0] > split_val]
            sub_df1 = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['lower_bound'] <= split_val, :]
            sub_df2 = pd_num_freq_tbl1.loc[pd_num_freq_tbl1['lower_bound'] > split_val, :]
            max_t_value = pd_num_freq_tbl1['abs_t_value'].max()
            
            if split_p > p_threshold:
                return
            if not split_val in cut_point_list:
                cut_point_list.append(split_val)
                t_value_list.append(max_t_value)
                p_value_list.append(split_p)
                if left_target_var_array.shape[0] >= int(self.df_samp_total_cnt*sub_population_pct):
                    self.LR_recursive_var_bin(sub_df1, left_target_var_array, p_threshold, sub_population_pct, cut_point_list, t_value_list, p_value_list)
                if right_target_var_array.shape[0] >= int(self.df_samp_total_cnt*sub_population_pct):
                    self.LR_recursive_var_bin(sub_df2, right_target_var_array, p_threshold, sub_population_pct, cut_point_list, t_value_list, p_value_list)
            else:
                return
        else:
            return
        
# update numerical variable's binning to reduce the number of total segments of each numercial variable while retaining as much R square of each numerical variable as much as possible for later linear regression
    
    def update_r_square_with_new_bin(self, p_threshold, sub_population_pct, selected_attributes = []):
        
        def bucketize(var,  var_name, quantile_dict):
            if var == None:
                return 'NULL - NULL'
            
            if var == -999:
                return '-999 - -999'
            
            quantiles = quantile_dict[var_name]['sorted_bin_seq']
            
            if len(quantiles) > 2:
                    if bisect(quantiles, var) <= len(quantiles) - 1:
                        higher_bound = quantiles[bisect(quantiles, var)]
                        lower_bound = quantiles[bisect(quantiles, var) - 1]
                    else:
                   
                        higher_bound = quantiles[len(quantiles) - 1]
                        lower_bound = quantiles[len(quantiles) - 1]
            else:
                    if bisect(quantiles, var) > 0:
                        higher_bound = quantiles[bisect(quantiles, var)-1]
                        lower_bound = quantiles[bisect(quantiles, var)-1]
                    else:
                        higher_bound = quantiles[0]
                        lower_bound = quantiles[0]
                        
                       

                    
            return str(lower_bound) + ' - ' + str(higher_bound)
        
        var_bin_dict = {}
        if len(selected_attributes) >= 1 and selected_attributes != None:
            pd_num_freq_tbl = self.init_r_square_info.loc[self.init_r_square_info['var_name'].isin(selected_attributes)]
        else:
            pd_num_freq_tbl = self.init_r_square_info
            
        #numerical attributes
        pd_num_freq_tbl0 = pd_num_freq_tbl.loc[pd_num_freq_tbl['var_name'].isin(self.num_var_quantile_dict.keys())]
        #categorical attributes
        pd_num_freq_tbl1 = pd_num_freq_tbl.loc[pd_num_freq_tbl['var_name'].isin(self.num_var_quantile_dict.keys()) == False]
        number_of_cat_var = pd_num_freq_tbl['var_name'].isin(self.num_var_quantile_dict.keys()) == False
        pd_num_freq_tbl0.replace(to_replace='NULL - NULL', value= '-999 - -999', inplace = True)
        
        for var in [str(x) for x in list(set(pd_num_freq_tbl0.var_name))]:
            if var in self.num_var_quantile_dict.keys():
                var_bin_dict.setdefault(var)
                var_bin_dict[var] = {}
                cut_point_list = []
                t_value_list = []
                p_value_list = []
                df = pd_num_freq_tbl0.loc[pd_num_freq_tbl0['var_name'] == var, :].dropna()
                target_var_array = np.array(self.df_samp.select([var, self.target_var]).sort(col(var)).collect())
                target_var_array = target_var_array[~(target_var_array[:, 0] == None)]
                if df.shape[0] == 1:
                    df.loc[:, 'lower_bound'] = df.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[0]))
                    cut_point_list.append(int(df.loc[:, 'lower_bound']))
                    t_value_list.append(0)
                    p_value_list.append(None)
                    var_bin_dict[var]['bin_seq'] = cut_point_list
                    var_bin_dict[var]['sorted_bin_seq'] = cut_point_list
                    var_bin_dict[var]['t_value_seq'] = t_value_list
                    var_bin_dict[var]['p_value_seq'] = p_value_list
                else:
                    df.loc[:, 'lower_bound'] = df.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[0]))
                    df.loc[:, 'higher_bound'] = df.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[1]))
                    self.LR_recursive_var_bin(df, target_var_array, p_threshold,sub_population_pct, cut_point_list, t_value_list, p_value_list)
                    
                    true_min_lower_bound = df.loc[:, 'lower_bound'].min()
                    true_max_higher_bound = df.loc[:, 'higher_bound'].max()
                    var_bin_dict[var]['bin_seq'] = cut_point_list
                    cut_point_list = cut_point_list + [true_min_lower_bound,true_max_higher_bound]
                    var_bin_dict[var]['sorted_bin_seq'] = sorted(list(set(cut_point_list)))
                    var_bin_dict[var]['t_value_seq'] = t_value_list
                    var_bin_dict[var]['p_value_seq'] = p_value_list
                    
        for var in var_bin_dict:
            var_bin_dict[var]['t_value_dict'] = dict(zip([str(x) for x in var_bin_dict[var]['bin_seq']],  var_bin_dict[var]['t_value_seq']))
            var_bin_dict[var]['binning_sequence'] = dict(zip([str(x) for x in var_bin_dict[var]['bin_seq']], range(len(var_bin_dict[var]['t_value_seq']))))
    
        self.var_bin_dict = var_bin_dict
        pd_num_freq_tbl0.loc[:, 'lower_bound'] = pd_num_freq_tbl0.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[0]))
        pd_num_freq_tbl0.loc[:, 'higher_bound'] = pd_num_freq_tbl0.loc[:, 'segment'].apply(lambda row: float(row.split(' - ')[1]))
        pd_num_freq_tbl0['new_bin'] = pd_num_freq_tbl0.apply(lambda row: bucketize(row['lower_bound'], row['var_name'], var_bin_dict), axis = 1)            
        if number_of_cat_var.sum() > 0:
            pd_num_freq_tbl1['lower_bound'] = pd_num_freq_tbl1['segment'].apply(lambda x: x.split(' - ')[0])
            pd_num_freq_tbl1['higher_bound'] = pd_num_freq_tbl1['segment'].apply(lambda x: x.split(' - ')[1])
            pd_num_freq_tbl1['new_bin'] = pd_num_freq_tbl1['segment']
            pd_freq_tbl = pd.concat([pd_num_freq_tbl0, pd_num_freq_tbl1])
        else:
            pd_freq_tbl = pd_num_freq_tbl0
        #key information needed to compute new R square: sum of segment_expalined_var, sum of segment_total_var, sum of target_var_sum, sum of total_count, new mean target, new r_square
        pd_freq_tbl1 = pd_freq_tbl.groupby(['var_name', 'new_bin']).agg({'segment_total_var':'sum', 'segment_explained_var':'sum', 'target_var_sum':'sum', 'total_count':'sum'})
        pd_freq_tbl1.reset_index(inplace = True)
        pd_freq_tbl1['lower_bound'] = pd_freq_tbl1.apply(lambda row: row['new_bin'].split('-')[0].strip(), axis = 1)
        pd_freq_tbl1['higher_bound'] = pd_freq_tbl1.apply(lambda row: row['new_bin'].split('-')[1].strip(), axis = 1)
        pd_freq_tbl1['numeric_lower_bound'] = pd.to_numeric(pd_freq_tbl1['lower_bound'], errors ='coerce')
        pd_freq_tbl1['numeric_higher_bound'] = pd.to_numeric(pd_freq_tbl1['higher_bound'], errors ='coerce')
        pd_freq_tbl1['updated_Rsquare'] = pd_freq_tbl1.apply(lambda row: float(row['segment_explained_var'])/float(row['segment_total_var']) if row['segment_total_var'] > 0 else 0, axis = 1)
        pd_freq_tbl1['target_mean'] = pd_freq_tbl1.apply(lambda row: float(row['target_var_sum'])/float(row['total_count']) if row['total_count'] > 0 else None, axis = 1)
        overall_r_stats = pd_freq_tbl.groupby(['var_name']).agg({'segment_explained_var':'sum', 'segment_total_var':'sum'})
        overall_r_stats.reset_index(inplace = True)
        overall_r_stats['total_Rsquare'] = overall_r_stats.apply(lambda row: float(row['segment_explained_var'])/row['segment_total_var'], axis = 1)
        overall_r_stats.sort_values(by = ['total_Rsquare'], ascending = False, inplace = True)
        pd_freq_tbl2 = pd.merge(pd_freq_tbl1, overall_r_stats.loc[:,['var_name', 'total_Rsquare']], how='left', on=['var_name'])
        pd_freq_tbl2.loc[:, 'var_type'] = pd_freq_tbl2.apply(lambda row: 'numerical' if row['var_name'] in self.num_var_quantile_dict.keys() else 'categorical', axis = 1)
        pd_freq_tbl2.sort_values(by = ['var_name', 'numeric_lower_bound'], ascending = True, inplace = True)
        pd_freq_tbl2.loc[:, 'lower_bound'] = pd_freq_tbl2['lower_bound'].apply(lambda x: '-999' if x == '' else x)
        self.output_r_df = pd_freq_tbl2
        
        self.univariate_varimp = pd_freq_tbl2.loc[:, ['var_name', 'total_Rsquare']].drop_duplicates().sort_values(by = ['total_Rsquare'], ascending = False)

        self.final_var_list = list(self.output_r_df['var_name'].unique())     
        var_mean_dict = {}
        
        for variable in self.final_var_list:
            var_mean_dict.setdefault(variable)
            var_mean_dict[variable] = self.output_r_df.loc[self.output_r_df.var_name == variable,['new_bin','target_mean']].set_index('new_bin').to_dict()['target_mean']
            
        self.var_encoding_dict = var_mean_dict
        
        return pd_freq_tbl2
    
    
# transform original modeling dataset in spark dataframe with optimally binned segments' WOE / Mean value for later prediction
    
    def transform_original_data(self, train_ratio = 0.8, var_to_transform = []):
        from pyspark.sql.functions import col
        
        def bucketize(var,  var_name, quantile_dict, var_woe_dict):
            
            if var == None:
                return 'NULL'
            quantiles = quantile_dict[var_name]
            
            if len(quantiles) > 2:
                    if bisect(quantiles, var) <= len(quantiles) - 1:
                        higher_bound = quantiles[bisect(quantiles, var)]
                        lower_bound = quantiles[bisect(quantiles, var) - 1]
                    else:
                   
                        higher_bound = quantiles[len(quantiles) - 1]
                        lower_bound = quantiles[len(quantiles) - 2]
            else:
                    if bisect(quantiles, var) > 0:
                        higher_bound = quantiles[bisect(quantiles, var)-1]
                        lower_bound = quantiles[bisect(quantiles, var)-1]
                    else:
                        higher_bound = quantiles[0]
                        lower_bound = quantiles[0]

            woe_returned = var_woe_dict[var_name][str(lower_bound) + ' - ' + str(higher_bound)] if str(lower_bound) + ' - ' + str(higher_bound) in var_woe_dict[var_name] else 0
            return woe_returned

        def find_segment_udf(quantile_dict, var_woe_dict):
            return udf(lambda var, var_name: bucketize(var, var_name, quantile_dict, var_woe_dict), DoubleType())

        def simplify_categorical_udf(var_woe_dict):
            return udf(lambda var, var_name: var_woe_dict[var_name][str(var) + ' - ' + str(var)] if str(var) + ' - ' + str(var) in var_woe_dict[var_name] else 0, DoubleType())
        
        
        if len(var_to_transform) == 0 :
            selected_numerical_attr = [var for var in self.final_var_list if var in self.selected_numerical_attr]
            selected_categorical_attr = [var for var in self.final_var_list if var in self.selected_categorical_attr]
        else:
            selected_numerical_attr = [var for var in var_to_transform if var in self.selected_numerical_attr]
            selected_categorical_attr = [var for var in var_to_transform if var in self.selected_categorical_attr]
            
        num_var_bin_dict = {}
        transformed_var_list = []
        segmented_attribute_list = []

        if self.output_r_df.empty:
            for var in selected_numerical_attr:
                if list(set(self.output_iv_df[self.output_iv_df.index == var].var_type))[0] == 'numerical':
                    num_var_bin_dict.setdefault(var)
                    num_var_bin_dict[var] = sorted([float(x) for x in list(set(list(self.output_iv_df[self.output_iv_df.index == var].lower_bound) + list(self.output_iv_df[self.output_iv_df.index == var].higher_bound)))])
        else:
            for var in selected_numerical_attr:
                if list(set(self.output_r_df[self.output_r_df.var_name == var].var_type))[0] == 'numerical':
                    num_var_bin_dict.setdefault(var)
                    num_var_bin_dict[var] = sorted([float(x) for x in list(set(list(self.output_r_df[self.output_r_df.var_name == var].lower_bound) + list(self.output_r_df[self.output_r_df.var_name == var].higher_bound)))])
            
        for var in selected_numerical_attr:  
            segmented_var = 'binned_' + var
            segmented_attribute_list.append(segmented_var)
            binned_df_col = self.df_samp.select(var)
            binned_df_col = binned_df_col.withColumn(segmented_var, find_segment_udf(num_var_bin_dict, self.var_encoding_dict)(col(var).alias('var'),lit(var).alias('var_name')).alias(segmented_var))
            binned_df_col = binned_df_col.select([col(attr) for attr in binned_df_col.columns if not attr in [var]])
            transformed_var_list.append(self.hc.as_h2o_frame(binned_df_col.select(col(segmented_var).alias(var))))
        
        for var in selected_categorical_attr:              
            segmented_var = 'binned_' + var
            segmented_attribute_list.append(segmented_var)
            binned_df_col = self.df_samp.select(var)
            binned_df_col = binned_df_col.withColumn(segmented_var, simplify_categorical_udf(self.var_encoding_dict)(col(var).alias('var'), lit(var).alias('var_name')))
            binned_df_col = binned_df_col.select([col(attr) for attr in binned_df_col.columns if not attr in [var]])
            transformed_var_list.append(self.hc.as_h2o_frame(binned_df_col.select(col(segmented_var).alias(var))))
        
        #finally append target variable in the end
        transformed_var_list.append(self.hc.as_h2o_frame(self.df_samp.select(self.target_var)))
        combined_cols = transformed_var_list[0]
        for col in transformed_var_list[1:]:
            combined_cols = combined_cols.cbind(col)
    
        self.h2o_model_df = combined_cols
        
        self.h2o_train,self.h2o_test = self.h2o_model_df.split_frame(ratios=[train_ratio])
            

    
    
# iteratively filter outer H2O GLM modeling variables with p value higher than 5% so that all final retained variables are statistically significant
    
    def iterative_model_selection(self, enet_glm, excluded_var_list, linear_reg):
        
        from h2o.grid.grid_search import H2OGridSearch
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator
        
        target_var = self.target_var
        if linear_reg == True:
            family = 'gaussian'
            
        else:
            family = 'binomial'
            
        train = self.h2o_train
        test = self.h2o_test
            
        enet_coef = enet_glm._model_json['output']['coefficients_table'].as_data_frame()
        
        enet_selected_var = [var for var in enet_coef.loc[enet_coef['coefficients'] > 0].names if var not in  excluded_var_list + ['Intercept']]
        
        final_glm = H2OGeneralizedLinearEstimator(family = family, compute_p_values = True, remove_collinear_columns = True, standardize = True, missing_values_handling = "skip", lambda_ = 0, solver='IRLSM')
        
        final_glm.train(x = enet_selected_var, y = target_var, training_frame = train, validation_frame = test)
        
        coef_table = final_glm._model_json['output']['coefficients_table'].as_data_frame()
        
        init_coef_table_shape = coef_table.shape[0]
        
        selected_coef_table = coef_table.loc[coef_table['p_value'] < 0.05]
        
        selected_coef_table_shape = selected_coef_table.shape[0]
        
        final_selected_var = [var for var in list(selected_coef_table.names) if not var in  excluded_var_list + ['Intercept']]
        
        while selected_coef_table_shape < init_coef_table_shape:
            
            final_selected_var = [var for var in list(selected_coef_table.names) if not var in  excluded_var_list + ['Intercept']]
            
            final_glm = H2OGeneralizedLinearEstimator(family = family,compute_p_values = True, remove_collinear_columns = True, standardize = True, missing_values_handling = "skip", lambda_ = 0, solver='IRLSM', nfolds = 5)
            
            final_glm.train(x = final_selected_var, y = target_var, training_frame = train, validation_frame = test)
            
            coef_table = final_glm._model_json['output']['coefficients_table'].as_data_frame()
            
            init_coef_table_shape = coef_table.shape[0]
            
            selected_coef_table = coef_table.loc[coef_table['p_value'] < 0.05]
            
            selected_coef_table_shape = selected_coef_table.shape[0]
            
        final_coef_table = coef_table
        
        final_coef_table['abs_std_coef'] = final_coef_table['standardized_coefficients'].abs()
        
        final_coef_table = final_coef_table.sort_values(by = 'abs_std_coef', ascending = False)
        
        return final_glm, final_coef_table


# H2O GLM model building with option for elastic net regularization for optimal parameter search
# consolidate final model with variables selected from elastic net regression and re-fit the model with MLE method
# to make sure all variables are statistically significant
    
    def optimal_glm_tuning(self, excluded_var_list = [], linear_reg=True, elmnet_enabled=True):

        target_var = self.target_var

        if linear_reg == True:
            family = 'gaussian'
            performance_metric = 'RMSE'
            
        else:
            family = 'binomial'
            self.h2o_train[target_var] = self.h2o_train[target_var].asfactor()
            self.h2o_test[target_var] = self.h2o_test[target_var].asfactor()
            performance_metric = 'AUC'
    
        train = self.h2o_train
        test = self.h2o_test

        predictors = [var for var in train.columns if not var in excluded_var_list+ [target_var]]

        if elmnet_enabled == True:
            hyper_parameters = {'alpha': [0.01, 0.05, 0.1, 0.2]}

            search_criteria = {'strategy': "RandomDiscrete", 'seed': 42,
                    'stopping_metric': performance_metric, 
                    'stopping_tolerance': 0.01,
                    'stopping_rounds': 5,
                    'max_runtime_secs': 2500}

            asymp_g = H2OGridSearch(H2OGeneralizedLinearEstimator(family= family, lambda_search = True, standardize = True,lambda_min_ratio = 0.01, nfolds = 5), hyper_parameters, search_criteria=search_criteria)
            asymp_g.train(x=predictors,y=target_var, training_frame=train, validation_frame = test)
            sorted_asymp_g = asymp_g.get_grid(performance_metric, decreasing = True)
            best_asymp_g = sorted_asymp_g.models[0]
        else:               
            best_asymp_g = H2OGeneralizedLinearEstimator(family = family, compute_p_values = True, remove_collinear_columns = True, standardize = True, missing_values_handling = "skip", lambda_ = 0, solver='IRLSM', nfolds = 10)
            best_asymp_g.train(x = predictors, y = target_var, training_frame = train, validation_frame = test)
        
        finalized_glm, finalized_coef_table = self.iterative_model_selection(best_asymp_g,  excluded_var_list, linear_reg)
        
        self.optimal_glm = finalized_glm
        
        self.optimal_coef_tbl = finalized_coef_table
        

    
# Automate the EDA, initial variable binning, binning optimization, original dataset transformation with encoding & final GLM modeling
    
    def General_Linear_Model(self, data_path = '', infile_format = '', sep = ',', df =None, target_var=None, linear_reg=False, increment_percent=0.05,  create_sample=False, sample_cnt=50000, train_ratio=0.7, test_ratio=0.15, preselected_attributes=[], elmnet_enabled=False):

        self.import_data(data_path = data_path, infile_format = infile_format, sep = sep, df = df,  target_var = target_var, create_sample = create_sample, sample_cnt = sample_cnt, increment_percent = increment_percent, linear_reg = linear_reg)
        
        if len(preselected_attributes) != 0:
            numerical_attributes = [ var for var in self.datatype_dict['numerical_attributes'] if var in preselected_attributes]
            categorical_attributes = [var for var in self.datatype_dict['categorical_attributes'] if var in preselected_attributes]      
        else: 
            numerical_attributes = self.numerical_attr
            categorical_attributes = self.categorical_attr
            
            
        if linear_reg == False:
            self.init_woe_iv(numerical_attr = numerical_attributes, categorical_attr = categorical_attributes)
            self.update_iv_with_new_bin(p_threshold = 0.05, sub_population_pct = 0.05, selected_attributes = preselected_attributes)
            self.transform_original_data(train_ratio = train_ratio,  var_to_transform = preselected_attributes)
        else:
            self.init_r_square(numerical_attr = numerical_attributes, categorical_attr = categorical_attributes, excluded_var = [])
            self.update_r_square_with_new_bin(p_threshold = 0.05, sub_population_pct = 0.05, selected_attributes = preselected_attributes)
            self.transform_original_data(train_ratio = train_ratio, var_to_transform = preselected_attributes)

        self.optimal_glm_tuning(excluded_var_list = [], linear_reg = linear_reg, elmnet_enabled= elmnet_enabled)
        
        return self.optimal_glm, self.optimal_coef_tbl

    # alternative method to create benchmark for optimal performance through hyper-parameter tunned GBM model building
    
    def GBM_model_eda(self, file_path, target_var,excluded_var_list = [],  linear_reg = True,  train_ratio = 0.8, top_n_var = 50):

        self.h2o_df = h2o.import_file(path = file_path)
        response = target_var
        predictors = [var for var in self.h2o_df.columns if var not in list([response] + excluded_var_list)]
        ## use all other columns (except for the name & the response column ("survived")) as predictors

        if linear_reg == True:
            distribution = 'gaussian'
            performance_metric = 'RMSE'
        else:
            distribution = 'bernoulli'
            performance_metric = 'AUC'
            self.h2o_df[response] = self.h2o_df[response].asfactor()
        
        #split original data into train, valid and test dataset
        train, test = self.h2o_df.split_frame(ratios=[train_ratio],  seed=1234, destination_frames=['train.hex','test.hex'])

        ## Depth 10 is usually plenty of depth for most datasets, but you never know
        init_hyper_params = {'max_depth' : [x for x in range(3,11,2)]}

        #Build initial GBM Model
        gbm_grid = H2OGradientBoostingEstimator(
            ## more trees is better if the learning rate is small enough 
            ## here, use "more than enough" trees - we have early stopping
            ntrees=200,
            ## smaller learning rate is better
            ## since we have learning_rate_annealing, we can afford to start with a 
            #bigger learning rate
            learn_rate=0.05,
            ## learning rate annealing: learning_rate shrinks by 1% after every tree 
            ## (use 1.00 to disable, but then lower the learning_rate)
            learn_rate_annealing = 0.99,
            ## sample 80% of rows per tree
            sample_rate = 0.8,
            ## sample 80% of columns per split
            col_sample_rate = 0.8,
            ## fix a random number generator seed for reproducibility
            seed = 1234,
            ## score every 10 trees to make early stopping reproducible 
            #(it depends on the scoring interval)
            score_tree_interval = 10, 
            ## early stopping once the validation AUC doesn't improve by at least 0.01% for 
            #5 consecutive scoring events
            stopping_rounds = 5,
            stopping_metric = performance_metric,
            stopping_tolerance = 1e-4)

        #Build grid search with previously made GBM and hyper parameters
        grid = H2OGridSearch(gbm_grid,init_hyper_params,
                         grid_id = 'depth_grid',
                         search_criteria = {'strategy': "Cartesian"})


        #Train grid search
        grid.train(x=predictors, 
           y=response,
           training_frame = train,
           validation_frame = test)



        ## sort the grid models by decreasing AUC if classification problem
        if linear_reg == False:
            sorted_grid = grid.get_grid(sort_by= performance_metric,decreasing=True)
            
        else:
            sorted_grid = grid.get_grid(sort_by = performance_metric, decreasing=False)

        max_depths = sorted_grid.sorted_metric_table()['max_depth'][0:4]
        new_max = int(max_depths.max())
        new_min = int(max_depths.min())




        gbm_tuning_params = {
                #keep learning rate small
                'learn_rate': [0.01,0.05],
                # keep a narrow range of max depth
                'max_depth': list(range(new_min,new_max+1,2)),
                #stochastic GBM, row wise randomization
                'sample_rate': [0.4, 0.8],
                #stochastic GBM, column wise randomization
                'col_sample_rate': [0.4, 0.8],
                #number of trees
                'ntrees': [50, 100, 200]}



        gbm_final_grid = H2OGradientBoostingEstimator(distribution= 'auto',
                    ## more trees is better if the learning rate is small enough 
                    ## here, use "more than enough" trees - we have early stopping
                    ## ntrees=500,
                    ## smaller learning rate is better
                    ## since we have learning_rate_annealing, we can afford to start with a 
                    #  bigger learning rate
                    ## learn_rate=0.04,
                    ## learning rate annealing: learning_rate shrinks by 1% after every tree 
                    ## (use 1.00 to disable, but then lower the learning_rate)
                    learn_rate_annealing = 0.99,
                    ## score every 10 trees to make early stopping reproducible 
                    #(it depends on the scoring interval)
                    score_tree_interval = 10,
                    ## fix a random number generator seed for reproducibility
                    seed = 1234,
                    ## early stopping once the validation AUC doesn't improve by at least 0.01% for 
                    #5 consecutive scoring events
                    stopping_rounds = 3,
                    stopping_metric = performance_metric,
                    stopping_tolerance = 1e-4,
                    nfolds = 5)
        
        search_criteria_tune = {'strategy': 'RandomDiscrete', 'max_models': 20, 'seed': 1}
            
        #Build grid search with previously made GBM and hyper parameters
        final_grid1 = H2OGridSearch(model = gbm_final_grid, hyper_params = gbm_tuning_params, 
                                    grid_id = 'final_grid',
                                    search_criteria = search_criteria_tune)
        #Train grid search
        start_time = time.time()
        final_grid1.train(x=predictors, 
           y=response,
           ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
           max_runtime_secs = 3600, 
           training_frame = train,
           validation_frame = test)



        end_time = time.time()
        print('total time it takes to do the random search is : ', end_time - start_time)
    
        if linear_reg == True:
            #minimize RMSE for linear regression
            sorted_final_grid = final_grid1.get_grid(sort_by= performance_metric,decreasing=False)
        else:
            #maximize AUC for logistic regression
            sorted_final_grid = final_grid1.get_grid(sort_by= performance_metric,decreasing=True)

        gbm_best_model = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][0])
        gbm_best_model_performance = gbm_best_model.model_performance(test)
    
        #extract top 50 most important attributes for the GBM model
        gbm_varimp = gbm_best_model.varimp(use_pandas = True)
        top_gbm_varimp = gbm_varimp.loc[gbm_varimp.index <= top_n_var]
    
        get_ipython().run_line_magic('matplotlib', 'inline')


        top_gbm_varimp.plot.bar(x='variable', y='scaled_importance')
        
        #extract optimal GBM model parameters
        optimal_GBM_parameters = {'optimal_learn_rate': gbm_best_model.get_params()['learn_rate']['actual_value'],
                                  'optimal_max_depth': gbm_best_model.get_params()['max _depth']['actual_value'],
                                  'optimal_ntrees': gbm_best_model.get_params()['ntrees']['actual_value'],
                                 'optimal_sample_rate':gbm_best_model.get_params()['sample_rate']['actual_value'],
                                 'optimal_col_sample_rate':gbm_best_model.get_params()['col_sample_rate']['actual_value']}
        
        self.gbm_best_model = gbm_best_model
        self.gbm_best_model_performance = gbm_best_model_performance
        self.gbm_best_model_params = optimal_GBM_parameters
        self.gbm_top_attributes = top_gbm_varimp

        

