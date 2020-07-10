import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler   #数据标准化模块
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_decomposition import train
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression

#confusion_matrix 计算混淆矩阵的方法，传入的是结果的（标签）预测值与实际值
#recall_score计算召回率的方法，传入的是结果的（标签）预测值与实际值
from sklearn.metrics import confusion_matrix,recall_score,classification_report
import itertools
from imblearn.over_sampling import SMOTE

oversampler=SMOTE(random_state=0)


data=pd.read_csv('creditcard.csv')
print(data.head())
print(Counter(data['Class']))  #Count函数统计numpy数组中各个值的个数
data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Time','Amount'],axis=1)
print(data.head())

#分离数据与标签
#数据
X=data.loc[:,data.columns!='Class']
print(X.head())
#标签
y=data.loc[:,data.columns=='Class']
print(y.head())

#获取异常样本数量
# number_records_fraud=len(data[data['Class']==1])
# print(number_records_fraud)

#获取异常记录的索引和异常数据的数量
fraud_indices=np.array(data[data['Class']==1].index)
print(fraud_indices)
number_records_fraud=len(fraud_indices)
print(number_records_fraud)

normal_indices=data[data['Class']==0].index
print(normal_indices)

#在正常样本中，随机采样出指定数量的样本，并取出其索引
#choise方法：从normal_indices中随机抽取number_records_fraud个数，replase=False，不能重复抽取
random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
print(random_normal_indices)

#根据索引值，重新将数据拼接到一起，把异常数据和正常数据重新拼接，拼接后要重新取样
#首先拼接索引值
under_samplt_indices=np.concatenate([fraud_indices,random_normal_indices])
print(under_samplt_indices)

#根据索引值获得下采样的数据
under_samplt_data=data.loc[under_samplt_indices,:]
print(under_samplt_data)

#提取数据和标签
#数据
X_undersample=under_samplt_data.loc[:,under_samplt_data.columns!='Class']
print(X_undersample)
# print("xxxxxxxxxxxxxxxxx")
# print(len(X_undersample))
#标签
y_undersample=under_samplt_data.loc[:,under_samplt_data.columns=='Class']
print(y_undersample)

print("正常样本所占的比例: ",len(under_samplt_data[under_samplt_data['Class']==0])/len(under_samplt_data))
print("异常样本所占的比例: ",len(under_samplt_data[under_samplt_data['Class']==1])/len(under_samplt_data))
print("样本总数： ",len(under_samplt_data))

#数据划分，一部分训练集，一部分测试集（验证集）
#输入参数分别是数据和标签
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
print("原始训练集包含样本数量: ",len(X_train))
print("原始测试集样本数量: ",len(X_test))
print("原始样本总数: ",len(X_test)+len(X_train))

#对下采样数据进行划分
X_train_undersample,X_test_undersample,y_train_undersample,y_test_undersamplt=train_test_split(X_undersample,y_undersample,test_size=0.3,random_state=0)
print("下采样训练集包含的样本数量: ",len(X_train_undersample))
print("下采样后的测试集数量: ",len(X_test_undersample))
print("下采样后的数据总数： ",len(X_train_undersample)+len(X_test_undersample))


def printing_Kfold_scores(x_train_data,y_train_data):
    fold=KFold(n_splits=5,shuffle=False)
    c_param_range=[0.01,0.1,1,10,100]
    results_table=pd.DataFrame(index=range(len(c_param_range)),columns=['C_parameter','Mean recall score'])
    j=0
    for c_param in c_param_range:
        print("-----------------------------")
        print("正则化惩罚力度: ",c_param)
        print("------------------------------")
        print("\n")
        recall_accs=[]

        #交叉验证
        for iteration,indices in enumerate(fold.split(x_train_data)):
            print("第%s次迭代" %iteration)
            lr=LogisticRegression(C=c_param,penalty='l1',solver='liblinear')
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())
            y_pred_undersample=lr.predict(x_train_data.iloc[indices[1],:])
            recall_acc=recall_score(y_train_data.iloc[indices[1],:],y_pred_undersample)
            print("第%s次迭代的召回率%s" %(iteration,recall_acc))
            recall_accs.append(recall_acc)
        print("平均召回率： ",np.mean(recall_accs))
        # results_table.loc[j, 'C_parameter'] = c_param
        # results_table.loc[j,'Mean recall score']=np.mean(recall_accs)
        results_table.loc[j,'C_parameter':'Mean recall score']=c_param,np.mean(recall_accs)
        j=j+1
    print(results_table)
    best_c=results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']
    print("最优解是: ",best_c)
    return best_c

best_c=printing_Kfold_scores(X_train_undersample,y_train_undersample)

#绘制混淆矩阵的方法
#cm计算
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure()
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

lr=LogisticRegression(C=0.01,penalty='l1',solver='liblinear')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred=lr.predict(X_test)

#混淆矩阵
cnf_matrix=confusion_matrix(y_test,y_pred)
#np.set_printoptions(precision=2)
class_names=[0,1]
plot_confusion_matrix(cnf_matrix,classes=class_names,title="Confusion matrix")
