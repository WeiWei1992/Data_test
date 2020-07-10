import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np
from sklearn.model_selection import train_test_split
#导入随机森林
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import pydot

# features=pd.read_csv('temps_extended.csv')
# print(features.head())
# features=features.drop(['ws_1','prcp_1','prcp_1','snwd_1'],axis=1)
# print(features.head())
# print("数据维度： ",features.shape)
# print(features.describe())
#
# years=features['year']
# months=features['month']
# days=features['day']
#
# dates=[str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
# dates=[datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# print(dates[:5])

# fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(10,10))
# fig.autofmt_xdate(rotation=45)
#
# #标签值
# ax1.plot(dates,features['actual'])
# ax1.set_xlabel('')
# ax1.set_ylabel('Temperature')
# ax1.set_title('Max Temp')
#
# #昨天的最高温度值
# ax2.plot(dates,features['temp_1'])
# ax2.set_xlabel('')
# ax2.set_ylabel('Temperature')
# ax2.set_title('Previout Max Temp')
#
# #前天的最高温度值
# ax3.plot(dates,features['temp_2'])
# ax3.set_xlabel('Date')
# ax3.set_ylabel('Temperature')
# ax3.set_title('Two Days Prior Max Temp')
#
# ax4.plot(dates,features['friend'])
# ax4.set_xlabel('Date')
# ax4.set_ylabel('Temperature')
# ax4.set_title('Friend Estimate')
# #plt.tight_layout(pad=2)
# plt.show()

#独热编码

# features=pd.get_dummies(features)
# print(features.head())
#
# #数据与标签
# #标签
# labels=np.array(features['actual'])
# #在特征中去掉标签
# features=features.drop('actual',axis=1)
# print(type(features))
# print(features)
# #名字单独保持，以备后患
# features_list=list(features.columns)
# # features=np.array(features)
# # print(type(features))
# # print(features)
#
# #数据集切分
# train_features,test_feature,train_labels,test_labels=train_test_split(features,labels,test_size=0.25,random_state=42)
# print("训练集特征： ",train_features.shape)
# print("训练集标签: ",train_labels.shape)
# print("测试集特征;  ",test_feature.shape)
# print("测试集标签： ",test_labels.shape)
#
# #建模
# rf=RandomForestRegressor(n_estimators=1000,random_state=42)
#
# #训练
# rf.fit(train_features,train_labels)
#
# #预测结果
# predictions=rf.predict(test_feature)
#
# #计算误差
# errors=abs(predictions-test_labels)
# mape=100*(errors/test_labels)
# print("MAPE: ",np.mean(mape))

# tree=rf.estimators_[5]
# export_graphviz(tree,out_file='tree.dot',feature_names=features_list,rounded=True,precision=1)
# (graph,)=pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')
#
#
# rf_small=RandomForestRegressor(n_estimators=10,max_depth=3,random_state=42)
# rf_small.fit(train_features,train_labels)
# tree_small=rf_small.estimators_[5]
# export_graphviz(tree_small,out_file='small_tree.dot',feature_names=features_list,rounded=True,precision=1)
# (graph,)=pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('small_tree.png')
# print("=================")

# importances=list(rf.feature_importances_)
# feature_importances=[(feature,round(importance,2)) for feature,importance in zip(features_list,importances)]
# feature_importances=sorted(feature_importances,key=lambda x:x[1],reverse=True)
# [print('Variable:{:20} Importances: {}'.format(*pair)) for pair in feature_importances]
# x_values=list(range(len(importances)))
#
# plt.bar(x_values,importances,orientation='vertical')
# plt.show()




features=pd.read_csv('temps_extended.csv')
print(features.head())
print("数据规模： ",features.shape)


# fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(15,10))
# fig.autofmt_xdate(rotation=45)
#
#
# #平均最高气温
# ax1.plot(features['average'])
# ax1.set_xlabel('')
# ax1.set_ylabel('Temperature (F)')
# ax1.set_title('Historical Avg Max Temp')
#
# ax2.plot(features['ws_1'],'r-')
# ax2.set_xlabel('')
# ax2.set_ylabel('Wind Speed (mph)')
# ax2.set_title('Prior Wind Speed')
#
# #降水
# ax3.plot(features['prcp_1'],'r-')
# ax3.set_xlabel('Date')
# ax3.set_ylabel("Precipitation (in)")
# ax3.set_title('Prior Precipitation')
#
# #积雪
# ax4.plot(features['snwd_1'],'ro')
# ax4.set_xlabel('Date')
# ax4.set_ylabel('Snow Depth (in)')
# ax4.set_title('Prior Snow Depth')
#
# plt.tight_layout(pad=2)
#
# plt.show()


#创建一个季节变量
seasons=[]
for month in features['month']:
    if month in [1,2,12]:
        seasons.append('winter')
    elif month in [3,4,5]:
        seasons.append('spring')
    elif month in [9,10,11]:
        seasons.append('fall')



