# coding=gbk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_decomposition import train
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,classification_report
import itertools
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)


data=pd.read_csv("creditcard.csv")
print(data.head())
count_class=pd.value_counts(data['Class'],sort=True)
print(count_class)
data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Time','Amount'],axis=1)
print(data.head())

X=data.iloc[:,data.columns!='Class']
print(X.head())
y=data.iloc[:,data.columns=='Class']
print(y.head())
number_records_fraud=len(data[data.Class==1])
print(number_records_fraud)


fraud_indices=np.array(data[data.Class==1].index)
print(fraud_indices)


normal_indices=data[data.Class==0].index
print(normal_indices)

random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
print(random_normal_indices)
print(len(random_normal_indices))
random_normal_indices=np.array(random_normal_indices)

under_samplt_indices=np.concatenate([fraud_indices,random_normal_indices])
print(under_samplt_indices)
print(len(under_samplt_indices))

under_samplt_data=data.iloc[under_samplt_indices,:]
print(under_samplt_data.head())

X_undersample=under_samplt_data.iloc[:,under_samplt_data.columns!='Class']
print(X_undersample.head())
y_undersample=under_samplt_data.iloc[:,under_samplt_data.columns=='Class']
print(y_undersample.head())

print(len(under_samplt_data[under_samplt_data.Class==0])/len(under_samplt_data))
print(len(under_samplt_data[under_samplt_data.Class==1])/len(under_samplt_data))
print(len(under_samplt_data))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
print("len(X_train): ",len(X_train))
print("len(X_test): ",len(X_test))
print("len(X_train)+len(X_test) :",len(X_train)+len(X_test))

X_train_undersample,X_test_undersample,y_train_undersample,y_test_undersample=train_test_split(X_undersample,y_undersample,test_size=0.3,random_state=0)
print('下采样训练集包含的样本数量',len(X_train_undersample))
print(len(X_test_undersample))
print(len(X_train_undersample)+len(X_test_undersample))

def printing_Kfold_scores(x_train_data,y_train_data):
    #fold=KFold(5,shuffle=False)
    fold = KFold(5,shuffle=False)

    print(fold)
    #定义不同的正则化惩罚力度
    c_param_range=[0.01,0.1,1,10,100]
    index = range(len(c_param_range), 2)
    print(index)
    results_table=pd.DataFrame(index=range(len(c_param_range)),columns=['C_parameter','Mean recall score'])
    results_table['C_parameter']=c_param_range
    print("++++++++++++++++++++++++++")
    print(results_table)

    j=0
    for c_param in c_param_range:
        print("-------------------------------------")
        print("正则化惩罚力度： ",c_param)
        print('--------------------------------------')
        print('')
        recall_accs=[]
        print("xxxxxxxxxxxxxxxxxx")
        print(fold)
        print(type(fold))
        #for iteration, indices in enumerate(fold,start=1):
        for iteration, indices in enumerate(fold.split(x_train_data)):
            #indices[0]训练集的索引
            #indices[1]测试集的索引

            print("===========================")
            # print(iteration)
            # print("indices[0]   :",indices[0])
            # print("len(indices[0]): ",len(indices[0]))
            # print("indices[1]   :",indices[1])
            # print("len(indices[1]) : ",len(indices[1]))
            print("==============================")
            lr=LogisticRegression(C=c_param,penalty='l1',solver='liblinear')
            lr.fit(x_train_data.iloc[indices[0],:].values,y_train_data.iloc[indices[0],:].values.ravel())

            y_pred_undersample=lr.predict(x_train_data.iloc[indices[1],:].values)
            recall_acc=recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Interation ',iteration,':召回率 = ',recall_acc)
        #当执行完成后计算平均结果
        results_table.loc[j,'Mean recall score']=np.mean(recall_accs)
        print(results_table)
        j+=1
        print("平均召回率: ",np.mean(recall_accs))

    #best_c=results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']
    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']
    print(results_table)

    print("*********************************************")
    print("效果最好的模型所选参数 = ",best_c)
    print("*****************************************")
    return best_c



def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#best_c=printing_Kfold_scores(X_train_undersample,y_train_undersample)

#
# lr = LogisticRegression(C = best_c, penalty = 'l1',solver='liblinear')
# lr.fit(X_train_undersample,y_train_undersample.values.ravel())
# y_pred_undersample = lr.predict(X_test_undersample.values)
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
# np.set_printoptions(precision=2)
# print("cnf_matrix:  \n")
# print(cnf_matrix)
#
# print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
#
# # Plot non-normalized confusion matrix
# class_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix
#                       , classes=class_names
#                       , title='Confusion matrix')
# plt.show()

# lr=LogisticRegression(C=0.01,penalty='l1',solver='liblinear')
# lr.fit(X_train_undersample,y_train_undersample.values.ravel())
# y_pred_undersamplt_proba=lr.predict_proba(X_test_undersample)
# # print(y_pred_undersamplt_proba)
# #指定不同的阈值
# thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# plt.figure(figsize=(10,10))
# j=1
# for i in thresholds:
#     y_test_predictions_high_recall=y_pred_undersamplt_proba[:,1]>i
#     print("pppppppppppppppppppppppppppppppppp")
#     print(y_test_predictions_high_recall)
#     plt.subplot(3,3,j)
#     j+=1
#     cnf_matrix=confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
#     print("Recall metric in the testing dataset: ",cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
#     class_names=[0,1]
#     plot_confusion_matrix(cnf_matrix,classes=class_names,title='Threshold > = %s'%i)
# plt.show()

print("========================================\n")
print("========================================\n")

credit_cards=pd.read_csv('creditcard.csv')
columns=credit_cards.columns
print(columns)
features_columns=columns.delete(len(columns)-1)
print(features_columns)
features=credit_cards[features_columns]
labels=credit_cards['Class']

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)
oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
print(len(os_labels[os_labels==1]))
print(len(os_labels[os_labels==0]))

print(os_features)
print(type(os_features))
print(os_labels)
print(type(os_labels))
os_labels=pd.DataFrame(os_labels)
print(os_labels)
print(type(os_labels))
# best_c=printing_Kfold_scores(os_features,os_labels)
# print(best_c)

lr = LogisticRegression(C = 10, penalty = 'l1',solver='liblinear')
lr.fit(os_features,os_labels.values.ravel())
y_pred = lr.predict(features_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()