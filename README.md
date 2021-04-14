# Create_Risk_Control_Rules

- [Create_Risk_Control_Rules](#create_risk_control_rules)
  - [1 Parameter conf](#1-parameter-conf)
  - [2 Data preprocessing](#2-data-preprocessing)
  - [3 Model](#3-model)
  - [4 Visualization](#4-visualization)
  - [5 Feature importance](#5-feature-importance)
  - [6 Rule extraction](#6-rule-extraction)
  - [7 OOT](#7-oot)


## 1 Parameter conf   
```python
### 1.1 model parameter configuration
# select tree models, the options available are'cart','rf','lgb','xgb','id3'\
tree_type = 'lgb'
# data split(needn't to do it in advance)，select a value between 0 and 1. When the value is equl to 0, it means that all data is train data.
test_size = 0.2
# flag columns
y_columns = 'overdueday_flag_7'
# Sensitive words (fuzzy match to remove data from x_columns)
ignore_word = ['overdueday_flag_7', 'usermobile', 'overdue_day_7_flag']
# Whether OOT is required, set to True if required, otherwise set to False
is_oot = False
# tree depth
max_depth = 5
# learning_rate, smaller, slower
learning_rate = 0.01
# The main purpose is to prevent model overfitting. preferably 'balanced' or None
class_weight = 'balanced'

# tree number
n_estimators = 10

### 1.2 Set path
# train data path
train_data_path = r'./raw_data/final_data_test.csv'
# oot data path
checking_data_path = r'./raw_data/final_data_test.csv'
### 1.3 Set save pth
# First level path to save (please add / at the end)
sav_path = './result/'

## I recommend that you do not change the content of the address below.
# x_columns is below and can be configured by yourself, if you don't configure it yourself, it is the remaining fields after deleting y_columns and the ignore_word that can be fuzzy matched,
# If you do not configure it yourself, you can delete the y_columns and the ignore_word that can be fuzzy matched, and then change the x_column below and set the ignore_word to [].
# picture save path
picture_path = sav_path+tree_type+'/'+'plot/'
# model save path
model_path = sav_path+tree_type+'/'+'model/'
# feature importance save path
importance_path = sav_path+tree_type+'/'+'importance/'
# rule save path
rule_path = sav_path+tree_type+'/'+'rule/'
# result of OOT save path
oot_path = sav_path+tree_type+'/'+'oot/'
```
## 2 Data preprocessing
```python
df = pd.read_csv(train_data_path)
# x_columns
x_columns = df.columns.tolist()
x_columns.remove(y_columns)
# initialize
data_pre = DataPreProcessing()
x_columns = data_pre.ignore_func(x_columns, ignore_word)
# missing value handling
# null_function, when the fourth parameter is 0, no missing values are processed and the missing values are uniformly processed as np.nan
new_data, x_columns_new = data_pre.null_function(df, x_columns, 1, 1, ignore_columns=['overdue_day_7_flag'], is_delete=0.8, null_value=[-999, '-999', '\\N', 'NULL'])
# Data filtering by iv value
# iv_threshold is the range of iv values, also use columns_num, take the top columns_num of IV values (from largest to smallest)
# return, the first is list, the result of the selected columns, the second is df, the corresponding iv value of all dictionaries
x_columns_final, df_iv = data_pre.select_columns(new_data, x_columns_new, y_columns, iv_threshold=[0.04, 0.5], columns_num=None)
```
## 3 Model 
```python
# initialize
model_ = Model()
# predict
model_final = model_.train_model(tree_type, new_data, x_columns_final, y_columns, max_depth, learning_rate,
                                 n_estimators, class_weight, test_size)
# model saving
model_.save_model(model_final, model_path)

# model lading
model_path = model_path+'model.pkl'
new_model = model_.load_model(model_path)
```
## 4 Visualization
```python
# initialize
# !!!Takes longer.
class_names = list(set(new_data[y_columns].astype('str')))
visual_ = Visualization()
visual_.plot(new_model, tree_type, picture_path, x_columns_final, class_names)
```

## 5 Feature importance 
```python
# initialize
importance_ = FeatureImportance()
importance_.importance(new_model, x_columns_final, importance_path)
```

## 6 Rule extraction
```python
# initialize
# The conclude_max parameter sets the level of rules to be brought up, for example, if the tree depth is set to 10, the first N rules can be extracted, and None is all.
# new_model model
# rule_path The path where the rules are stored after extraction, rule_path must be an xlsx file
conclude_max = None
rule_ = GetRule()
rule_.get_rule(tree_type, new_model, new_data, x_columns_final, y_columns, conclude_max, rule_path)
```
## 7 OOT 
```python
# There are 8 conditions for oot filtering, the condition will be filtered automatically according to the conditions set, and the remaining rules will be sent to oot for evaluation.
# If you want to filter by yourself, change the rule_path after you delete the selection and set select_condition to []
# If the condition_select is 1 < x < 2, divide it into two conditions, i.e. ['x > 1', 'x > 2'] Note: Leave a space before and after the symbol, a ! space!!!


# Initialize the time external verification module
# new_data temporal outlier data, customizable
# x_columns_final feature name
# y_columns target column names
# rule_path_new xlsx sheet names must be bad and good
# oot_path The path where the results of the out-of-time checks are saved
# es_sample_rate,es_sample_num,es_real_bad,es_real_good,es_real_bad_rate,es_real_good_rate,es_auc,es_recall
rule_path_new = rule_path + 'rule.xlsx'
condition_select = ['es_real_bad_rate > 0', 'es_real_good_rate > 0']
if is_oot:
    oot_ = OOT()
    oot_.oot(checking_data_path, x_columns_final, y_columns, rule_path_new, oot_path, condition_select)
```
