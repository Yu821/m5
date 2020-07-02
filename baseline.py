import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import lightgbm as lgb
#import dask_xgboost as xgb
#import dask.dataframe as dd
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import gc
import os
from scipy.sparse import csr_matrix
import sys
	
def reduce_mem_usage(df, verbose=True):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2    
	for col in df.columns: #columns毎に処理
		col_type = df[col].dtypes
		if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)    
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df

import IPython

# Reference: https://www.kaggle.com/girmdshinsei/for-japanese-beginner-with-wrmsse-in-lgbm
def weight_calc(data,product,lower_date,upper_date):

	# calculate the denominator of RMSSE, and calculate the weight base on sales amount

	sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

	d_name = ['d_' + str(i+1) for i in range(1913)]

	sales_train_val = weight_mat_csr * sales_train_val[d_name].values

	# calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
	# 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
	df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))

	start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1


	# denominator of RMSSE / RMSSEの分母
	weight1 = np.sum((np.diff(sales_train_val,axis=1)**2),axis=1)/(1913-start_no)

	# calculate the sales amount for each item/level
	df_tmp = data[(data['date'] > lower_date) & (data['date'] <= upper_date)]
	df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
	df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum).values

	weight2 = weight_mat_csr * df_tmp 

	weight2 = weight2/np.sum(weight2)

	del sales_train_val
	gc.collect()

	return weight1, weight2

def wrmsse(preds, data):

	# actual obserbed values / 正解ラベル
	y_true = data.get_label()

	# number of columns
	num_col = len(y_true)//NUM_ITEMS

	# reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
	reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
	reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T

	x_name = ['pred_' + str(i) for i in range(num_col)]
	x_name2 = ["act_" + str(i) for i in range(num_col)]

	train = np.array(weight_mat_csr*np.c_[reshaped_preds, reshaped_true])

	score = np.sum(
				np.sqrt(
					np.mean(
						np.square(
							train[:,:num_col] - train[:,num_col:])
						,axis=1) / weight1) * weight2)

	return 'wrmsse', score, False


def encode_categorical(df, cols):
	
	for col in cols:
		# Leave NaN as it is.
		le = LabelEncoder()
		not_null = df[col][df[col].notnull()]
		df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

	return df


def pca_fe(data):
	
	lag_cols = [col for col in data.columns.values if 'rolling_mean' in col]
	
	pca = PCA(n_components=3)
	pca_data = pca.fit_transform(data[lag_cols])
	
	data = pd.concat([data,pca_data],axis=1)
	
	return data
	


def time_fe(data):
	dt_col = 'date'
	
	data[dt_col] = pd.to_datetime(data[dt_col])
	attrs = [
#         "year",
#         "quarter",
#         "month",
        "week",
        "day",
		"dayofweek",
	]

	for attr in attrs:
		dtype = np.int16 if attr == "year" else np.int8
		data[attr] = getattr(data[dt_col].dt, attr).astype(dtype)

#     data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)

	return data

def lag_fe(data,idx=28):

	for size in [7,28,56,84,112]:
		data[f"rolling_mean_t{size}"] = data.groupby(["id"])["demand"].transform(
			lambda x: x.shift(idx).rolling(size).mean()
		)
        
	for size in [28,84]:
		data[f"rolling_std_t{size}"] = data.groupby(["id"])["demand"].transform(
			lambda x: x.shift(idx).rolling(size).std()
		)


	for size in [4,8,12]:
		data[f"dayofweek_rolling_mean_t{size}"] = data.groupby(["id","dayofweek"])["demand"].transform(
			lambda x: x.shift(idx).rolling(size).mean()
		)
	
	# Probability to get 0 demand
	data.loc[data[data['demand']==np.nan].index,'demand'] = -1
	data.loc[data[data['demand']==0].index,'demand'] = np.nan

	for size in [365]:
		data[f"zero_prob_t{size}"] = data.groupby(["id"])["demand"].transform(
			lambda x: x.shift(idx).rolling(size).count() / size
		)

	data['demand'] = data['demand'].fillna(0)
	data.loc[data[data['demand']==-1].index,'demand'] = np.nan

	return data


print('Reading files...')

submission = pd.read_feather("../input/reduce-data/sample_submission.feather")

# 予測期間とitem数の定義
NUM_ITEMS = 30490  # 30490
DAYS_PRED = 28  # 28


data = pd.read_feather("../input/encoded-combined-data/combined_data.feather")
product = pd.read_feather("../input/encoded-combined-data/product.feather")


data = data[data['date']>'2015-01-01']


tr_lower_date = '2016-03-27'
tr_upper_date = '2016-04-24'

lower_date = '2016-04-24'
upper_date = '2016-05-22'


print('\n Calculating weight ----------------- \n')

weight_mat = np.c_[np.identity(NUM_ITEMS).astype(np.int8), #item :level 12
				   np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
				   pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
				   pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values
				   ].T

weight_mat_csr = csr_matrix(weight_mat)
del weight_mat; gc.collect()

weight1, weight2 = weight_calc(data,product,lower_date,upper_date)


print('\n Making features ----------------- \n')

data = time_fe(data)
data = lag_fe(data)
data = reduce_mem_usage(data)

# 2/29
drop_feats = ['demand','index','id','part','date','wm_yr_wk','event_name_2','event_type_2']

# going to evaluate with the last 28 days -----------------------------------------------------------------------------------------------
y_train = data[(data['date'] > tr_lower_date) & (data['date'] <= tr_upper_date)]['demand']
x_train = data[(data['date'] > tr_lower_date) & (data['date'] <= tr_upper_date)].drop(drop_feats,axis=1)
y_val = data[(data['date'] > lower_date) & (data['date'] <= upper_date)]['demand']

val_id_date = data[(data['date'] > lower_date) & (data['date'] <= upper_date)][['id','date']]
x_val = data[(data['date'] > lower_date) & (data['date'] <= upper_date)].drop(drop_feats,axis=1)

test_id_date = data[(data['date'] > upper_date)][['id','date']]
test = data[(data['date'] > upper_date)].drop(drop_feats,axis=1)


print('\n Num features: ', x_train.shape[1])
print('\n Features: ', x_train.columns.values)

train_set = lgb.Dataset(x_train, y_train)
val_set = lgb.Dataset(x_val, y_val)

del x_train, y_train

params = {
    'boosting_type': 'gbdt',
    'objective': 'rmse',
    'learning_rate': 0.01,
    'num_leaves':75,
    'min_data_in_leaf':55,
    'n_jobs': -1,
    'metric': 'custom'
#     'bagging_fraction': 0.65,
#     'bagging_freq': 2, 
#     'feature_fraction': 0.9,
#     'colsample_bytree': 0.75
}


print('\n Training pre_model ----------------- \n')

# model estimation
pre_model = lgb.train(params, train_set, num_boost_round = 20000, early_stopping_rounds = 100,
                  valid_sets = [train_set, val_set], verbose_eval = 100, feval= wrmsse)

val_id_date['demand'] = pre_model.predict(x_val, num_iteration=pre_model.best_iteration)

val_score = np.sqrt(metrics.mean_squared_error(val_id_date['demand'], y_val))

print(f'Our val rmsse score is {val_score}')


y_train = data[(data['date'] > lower_date) & (data['date'] <= upper_date)]['demand']
x_train = data[(data['date'] > lower_date) & (data['date'] <= upper_date)].drop(drop_feats,axis=1)

train_set = lgb.Dataset(x_train, y_train)

print()
print('Best iteration: ', pre_model.best_iteration)
print()

print('\n Training model ----------------- \n')

model = lgb.train(params, train_set, num_boost_round = pre_model.best_iteration)

y_pred = model.predict(test)

test_id_date['demand'] = y_pred

# 	lgb.plot_importance(model,importance_type='gain',max_num_features=None)
fe_importance = pd.DataFrame()
fe_importance["feature"] = x_train.columns.values
fe_importance["importance"] = model.feature_importance(importance_type='gain')

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=fe_importance.sort_values(by='importance',ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

del data; gc.collect()


validation = val_id_date[['id', 'date', 'demand']]
validation = pd.pivot(validation, index = 'id', columns = 'date', values = 'demand').reset_index()
validation.columns = ['id'] + ['F' + str(i) for i in range(1,29)]
validation['id'] = validation['id'].str.replace(r'evaluation', 'validation')


evaluation = test_id_date[['id', 'date', 'demand']]
evaluation = pd.pivot(evaluation, index = 'id', columns = 'date', values = 'demand').reset_index()
evaluation.columns = ['id'] + ['F' + str(i) for i in range(1,29)]


validation = submission[['id']].merge(validation, on = 'id')
evaluation = submission[['id']].merge(evaluation, on = 'id')


final = pd.concat([validation, evaluation])

final.to_csv('new_sub.csv', index = False)
