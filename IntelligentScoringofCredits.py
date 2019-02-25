#用户：jasmineHuang

#日期：2019-02-24

#时间：14:21

#文件名称：PyCharm

import pandas as pd
import numpy as np
import pylab
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']
pylab.mpl.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns',None)

train = pd.read_csv('train_dataset.csv')
test = pd.read_csv('test_dataset.csv')

#EDA
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
sns.distplot(train['用户年龄'],bins=20,kde=False,hist_kws={'color':'steelblue'})
plt.subplot(2,2,2)
sns.distplot(train['用户网龄（月）'],bins=20,kde=False,hist_kws={'color':'blue'})
plt.subplot(2,2,3)
sns.distplot(train['缴费用户最近一次缴费金额（元）'],bins=20,kde=False,hist_kws={
    'color':'red'})
plt.subplot(2,2,4)
sns.distplot(train['当月通话交往圈人数'],bins=20,kde=False,hist_kws={'color':'green'})
plt.show()

#剔除用户年龄小于5的
train=train[train['用户年龄']>5]
sns.distplot(train['信用分'],bins=30)
plt.show()

#相关性分析
corr=train.corr()
plt.figure(figsize=(12,9))
k=10
cols=corr.nlargest(k,'信用分')['信用分'].index
print(cols)
cm=np.corrcoef(train[cols].values.T)
print(cm)
sns.set(font_scale=1.25)
sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={
    'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()

#删除高相关中的一个变量
train.drop('用户账单当月总费用（元）',axis=1,inplace=True)
test.drop('用户账单当月总费用（元）',axis=1,inplace=True)

#提取特征
def simple_features(df):
    df = df.copy()
    df['次数'] = df['当月网购类应用使用次数'] + df['当月物流快递类应用使用次数'] + df['当月金融理财类应用使用总次数'] + \
               df['当月视频播放类应用使用次数'] \
               + df['当月飞机类应用使用次数'] + df['当月火车类应用使用次数'] + df['当月旅游资讯类应用使用次数']
    df['交通工具类应用次数']=df['当月飞机类应用使用次数'] + df['当月火车类应用使用次数']

    for col in ['当月金融理财类应用使用总次数' , '当月旅游资讯类应用使用次数']:  # 这两个比较积极向上一点
        df[col + '百分比'] = df[col].values / (df['次数'].values+1)

    df['用户上网年龄百分比'] = df['用户网龄（月）'] / (df['用户年龄']+1)
    df['通话人均时长']=df['用户近6个月平均消费值（元）']/df['当月通话交往圈人数']
    df["是否不良客户"] = df["是否黑名单客户"] + df["是否4G不健康客户"]

    return df

train=simple_features(train)
test=simple_features(test)

#使用分段年龄
x1=train['用户年龄']
cut=np.arange(5,115,5)
x = np.array(x1)
bins = np.array(cut)
cut_age_labels = np.digitize(x , bins)
train['用户年龄已分段'] = cut_age_labels
t1=test['用户年龄']
t = np.array(t1)
t_cut_age_labels=np.digitize(t,bins)
test['用户年龄已分段'] = t_cut_age_labels

#使用通话人数分段
x2=train['当月通话交往圈人数']
cut=np.arange(0,1660,10)
x = np.array(x2)
bins = np.array(cut)
cut_tonghua_labels = np.digitize(x , bins)
train['当月通话交往圈人数已分段'] = cut_age_labels
t2=test['当月通话交往圈人数']
t = np.array(t2)
t_cut_age_labels=np.digitize(t,bins)
test['当月通话交往圈人数已分段'] = t_cut_age_labels

#获取训练和测试集
y_train=train['信用分']
train.drop(['信用分','用户编码'],axis=1,inplace=True)
x_train=train
submit=pd.DataFrame()
submit['id']=test['用户编码']
test.drop(['用户编码'],axis=1,inplace=True)
x_test=test
print(x_test.info())
print(x_test.describe())
print(x_test.isnull().sum().sort_values(ascending=False).head(10))

#用XGBoost进行训练和预测
import xgboost as xgb
#dataTrain = xgb.DMatrix(x_train,label=y_train)
#dataTest = xgb.DMatrix(x_test)
model = xgb.XGBRegressor()
model.fit(x_train,y_train)
submit['score'] = model.predict(x_test).astype('int')
submit.to_csv('submit_5.csv',index=False,encoding='utf-8')

#得分0.06155120000，提交时间:2019/02/24 15:11