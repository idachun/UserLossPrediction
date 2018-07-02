
# coding: utf-8

# ## 数据字段说明
# <br>
# 
#  | 变量名   | State  | Account Length | Area Code | Phone    | Int'l Plan       | VMail Plan 	   | VMail Message | Day Mins   | Day Calls    | Day Charge   | Eve Mins   | Eve Calls    | Eve Charge 	 | Night Mins | Night Calls  | Night Charge | Intl Mins  | Intl Calls   | Intl Charge  | CustServ Calls | Churn?         | 
#  | ------   | ------ | ------         | ------    | ------   | ------           | ------           | ------        | ------     | ------       | ------       | ------     | ------       | ------       | ------     | ------       | ------       | ------     | ------       | ------       | ------         | ------         |
#  | 变量解释 | 州     | 未知           | 区号      | 电话号码 | 是否为该计划用户 | 是否为该计划用户 | VMail消息量   | 每天通话量 | 白天通话次数 | 白天通话费用 | 傍晚通话量 | 傍晚通话次数 | 傍晚通话费用 | 夜间通话量 | 夜间通话次数 | 夜间通话费用 | 国际通话量 | 国际通话次数 | 国际通话费用 | 客服通话次数   | 是否为流失客户 | 
# 
# <br>
#  数据很简单,每行代表一个预订的电话用户。 每列包含客户属性，例如电话号码，在一天中不同时间使用的通话分钟，服务产生的费用，生命周期帐户持续时间以及客户是否仍然是客户。

# ## 改进日志
# - 0619,GDBT效果最好，未标准化
# recall score:0.7609 
# 
# - 0620，基于卡方分布、最大信息系数选择特征，删了一个特征，训练集GDBT精确率提高0.5%，随机森林模型召回率提高了3%，精确率提高了0.5%
# recall score:0.7609
# 
# - 0621,使用了好几种方式特征选择，有基于GDBT单变量特征选择、递归特征消除、基于L1正则化筛选特征、基于GDBT特征重要性特征选择，GDBT召回率提高了0.2%,最后对GDBT模型调参，召回率提高了1%
#     - precision score: 0.8740
#     - recall score:0.8043
# - 0625,使用smote算法过采样，召回率提高了2%
#     - precision score: 0.7808
#     - recall score:0.8261
# - 0626,使用聚类划分小簇和离群点得分来检测异常值，效果不好
#     - precision score: 0.8571
#     - recall score:0.7826
# - 0627,构造3个特征，删了9个特征，召回率提高了3%
#     - precision score: 0.9147
#     - recall score:0.8551
# - 0628,采用stacking方式融合模型rf和gdbt,召回率降了0.8%，精确率提高了5%
#     - precision score: 0.9669
#     - recall score:0.8478

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC


# In[113]:


data=pd.read_csv("../input/churn.csv")
print data.shape
data.head()


# In[114]:


# 查看特征列类型分布
data.dtypes.value_counts()


# In[115]:


# 查看具体哪些特征列是object类型
data.dtypes[data.dtypes=='object']


# In[116]:


# 删除无关特征
# data=data.drop(['State','Phone'],axis=1)
data=data.drop(['State','Area Code','Phone'],axis=1)


# In[117]:


# 类型转换
data['Churn?']=data['Churn?'].map(lambda x: 1 if x=='True.' else 0)
cols = ["Int'l Plan","VMail Plan"]
for col in cols:
    data[col]=data[col].map(lambda x: 1 if x=='yes' else 0)


# In[118]:


# 检查缺失值
data.isnull().sum().sort_values()


# In[119]:


#划分训练集和测试集
y=data.pop('Churn?')
train,test,train_y,test_y=train_test_split(data,y,test_size=0.3,random_state=0)


# In[120]:


# 评价函数
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
def train_cv(X,y,clf):
    print("precision score: %.4f"%cross_val_score(clf, X, y, cv=cv, scoring='precision').mean())
    print("recall score:%.4f "%cross_val_score(clf, X, y, cv=cv,scoring='recall').mean())


# In[10]:


train_cv(train,train_y,LR())
print
train_cv(train,train_y,SVC(random_state=0))
print
train_cv(train,train_y,KNN())
print
train_cv(train,train_y,RF(random_state=0))
print
train_cv(train,train_y,GBC(random_state=0))
print


# In[11]:


clf=GBC(random_state=0).fit(train,train_y)
pred=clf.predict(test)
metrics.recall_score(test_y,pred)


# ### 方差筛选

# In[12]:


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit(train)
np.sort(selector.variances_)


# In[13]:


for i in range(10,61,20):
    selector = VarianceThreshold(threshold=i/100)
    selector.fit(train)
    train1=selector.transform(train)

    train_cv(train1,train_y,GBC(random_state=0))
    print
    train_cv(train1,train_y,RF(random_state=0))
    print "____________________"


# ### 卡方检验

# In[14]:


from sklearn.feature_selection import chi2
corrlation={}
for i in range(train.shape[1]):
    corrlation[train.columns[i]]=chi2(train,train_y)[0][i]
pd.DataFrame.from_dict(corrlation,orient='index').sort_values(by=[0],ascending=False)


# In[15]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
score=0
index=1
for i in range(1,train.shape[1]+1):
    model=SelectKBest(chi2,k=i)
    a_train=model.fit_transform(train,train_y)
    
    clf = GBC(random_state=0)
    cv_score=cross_val_score(clf, a_train, train_y, cv=cv, scoring='recall').mean()
    if score<cv_score:
        score=cv_score
        index=i
    print i,round(cross_val_score(clf, a_train, train_y, cv=cv, scoring='precision').mean(),4),round(cv_score,4)
print "______________________"
print index,score
# 被删除的特征
model=SelectKBest(chi2,k=index).fit(train,train_y)
train.columns[~model.get_support()]


# ### 最大信息系数

# In[16]:


from minepy import MINE
m=MINE()
cols=train.columns
corrlation={}
for col in cols:
    m.compute_score(train[col],train_y)
    corrlation[col]=m.mic()
pd.DataFrame.from_dict(corrlation,orient='index').sort_values(by=[0],ascending=False)


# In[17]:


# 每次运行时出现的结果可能不同
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
score=0
index=1
for i in range(1,train.shape[1]+1):
    model=SelectKBest(mutual_info_classif,k=i)
    a_train=model.fit_transform(train,train_y)
    
    clf = GBC(random_state=0)
    cv_score=cross_val_score(clf, a_train, train_y, cv=cv, scoring='recall').mean()
    if score<cv_score:
        score=cv_score
        index=i
    print i,round(cross_val_score(clf, a_train, train_y, cv=cv, scoring='precision').mean(),4),round(cv_score,4)
print "______________________"
print index,score
# 被删除的特征
model=SelectKBest(mutual_info_classif,k=index).fit(train,train_y)
train.columns[~model.get_support()]


# ### 基于相关系数的假设检验

# In[18]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
score=0
index=1
for i in range(1,train.shape[1]+1):
    model=SelectKBest(f_classif,k=i)
    a_train=model.fit_transform(train,train_y)
    
    clf = GBC(random_state=0)
    cv_score=cross_val_score(clf, a_train, train_y, cv=cv, scoring='recall').mean()
    if score<cv_score:
        score=cv_score
        index=i
    print i,round(cross_val_score(clf, a_train, train_y, cv=cv, scoring='precision').mean(),4),round(cv_score,4)
print "______________________"
print index,score
model=SelectKBest(f_classif,k=index).fit(train,train_y)
train.columns[~model.get_support()]


# ### 基于GDBT的单变量特征选择

# In[19]:


clf =GBC(random_state=0)
scores=[]
columns=train.columns
corrlation={}
for i in range(train.shape[1]):
    score=cross_val_score(clf,train.values[:,i:i+1],train_y.reshape(-1,1),scoring='recall',
                          cv=cv)
    corrlation[columns[i]]=format(np.mean(score),'.4f')
pd.DataFrame.from_dict(corrlation,orient='index').sort_values(by=[0],ascending=False)


# In[20]:


# 删除特征重要性倒数第一的特征
train1=train.drop(['VMail Plan'],axis=1)
clf =GBC(random_state=0)
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# ### 递归特征消除

# In[21]:


# 通过交叉验证自动确定消除特征数目
from sklearn.feature_selection import RFECV

clf=RFECV(estimator=GBC(random_state=0),step=1,cv=cv,scoring='recall')
clf.fit(train,train_y)
# 被消除的特征
print train.columns[~clf.support_],np.max(clf.grid_scores_)
# 消除特征剩余个数对应得分
#clf.grid_scores_


# In[22]:


# 综合前面的选择，删除两个特征
train1=train.drop(['Night Calls','Account Length'],axis=1)
clf =GBC(random_state=0)
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# ### 基于L1的LR特征选择

# In[23]:


# 每次运行结果不一样
from sklearn.feature_selection import SelectFromModel

score=0
index=0
clf1=LR(penalty="l1").fit(train.values, train_y.values.reshape(-1,1))
for i in range(60,160,5):
    model = SelectFromModel(clf1,threshold=i/100000.0)
    model.fit(train,train_y)
    train1=model.transform(train)
    clf =GBC(random_state=0)
    cv_score=cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean()
    if score<cv_score:
        score=cv_score
        index=i/100000.0
    print i/100000.0,cv_score
print
print index,score


# In[24]:


clf1=LR(penalty="l1").fit(train.values, train_y.values.reshape(-1,1))
model = SelectFromModel(clf1,threshold=index)
model.fit(train,train_y)
train1=model.transform(train)

clf =GBC(random_state=0)
print train.columns[~model.get_support()]
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# ### 基于GDBT的特征选择

# In[25]:


from sklearn.feature_selection import SelectFromModel

score=0
index=0
clf1=GBC(random_state=0).fit(train.values, train_y.values.reshape(-1,1))
for i in range(250,350,10):
    model = SelectFromModel(clf1,threshold=i/10000.0)
    model.fit(train,train_y)
    train1=model.transform(train)
    clf =GBC(random_state=0)
    cv_score=cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean()
    if score<cv_score:
        score=cv_score
        index=i/10000.0
    print i/10000.0,cv_score
print
print index,score


# In[26]:


model = SelectFromModel(clf1,threshold=index)
model.fit(train,train_y)
train1=model.transform(train)

clf =GBC(random_state=0)
print train.columns[~model.get_support()]
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# #### 小结
# - 方差筛选
#     - discarded feature: None
#     - recall score:0.7362 
# - 卡方检验
#     - discarded feature: 'Eve Calls'
#     - recall score:0.7536231884057971
# - 互信息法
#     - discarded feature: 'Account Length', 'VMail Message', 'Eve Mins'
#     - recall score:0.7507246376811594
# - 基于相关系数的假设检验
#     - discarded feature: 'Eve Calls'
#     - recall score:0.7536231884057971
# - 基于GDBT的单变量特征选择
#     - discarded feature: 'VMail Plan'
#     - recall score:0.7391
# - 递归特征消除
#     - discarded feature: 'Night Calls','Account Length'
#     - recall score:0.7478
# - 基于L1的LR特征选择
#     - discarded feature: 'Account Length', 'Night Calls', 'Night Charge', 'Intl Charge'
#     - recall score:0.7507
# - 基于GDBT的特征选择
#     - discarded feature: 'Account Length', 'VMail Plan', 'Day Calls', 'Eve Calls'
#     - recall score:0.7507

# In[11]:


# 去除特征Eve Calls
train1=train.drop(['Eve Calls'],axis=1)
clf =GBC(random_state=0)
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# In[38]:


# 去除特征Eve Calls,Night Calls
train1=train.drop(['Eve Calls', 'Night Calls'],axis=1)
clf =GBC(random_state=0)
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# In[12]:


clf.fit(train1,train_y)
test1=test.drop('Eve Calls',axis=1)
pred=clf.predict(test1)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# In[40]:


clf.fit(train1,train_y)
test1=test.drop(['Eve Calls','Night Calls'],axis=1)
pred=clf.predict(test1)
metrics.recall_score(test_y,pred)


# ### GDBT调参

# In[41]:


from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(75,90,1)}
gsearch1 = GridSearchCV(estimator = GBC(learning_rate=0.1,max_depth=4,random_state=0), 
                   param_grid = param_test1, scoring='recall',iid=False,cv=cv)
gsearch1.fit(train1,train_y)
gsearch1.best_score_,gsearch1.best_params_


# In[42]:


#调节参数max_depth和min_samples_split
param_test2 = {'max_depth':range(3,9,2), 'min_samples_split':range(2,503,100)}
gsearch2 = GridSearchCV(estimator =GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
                                                          random_state=0), 
                    param_grid = param_test2, scoring='recall',iid=False, cv=cv)
gsearch2.fit(train1,train_y)
gsearch2.best_score_,gsearch2.best_params_


# In[43]:


#调节参数min_samples_split和min_samples_leaf
param_test3 = {'min_samples_split':range(2,200,50), 'min_samples_leaf':range(1,100,10)}
gsearch3 = GridSearchCV(estimator = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
                                                          max_depth=gsearch2.best_params_['max_depth'], random_state=0), 
                   param_grid = param_test3, scoring='recall',iid=False, cv=cv)
gsearch3.fit(train1,train_y)
gsearch3.best_score_,gsearch3.best_params_


# In[13]:


# clf = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
#                             max_depth=gsearch2.best_params_['max_depth'], 
#                             min_samples_leaf =gsearch3.best_params_['min_samples_leaf'],
#                             min_samples_split =gsearch3.best_params_['min_samples_split'],
#                         random_state=0)
clf = GBC(learning_rate=0.1, n_estimators=78,
                            max_depth=5, 
                            min_samples_leaf =1,
                            min_samples_split =52,
                        random_state=0)
round(cross_val_score(clf, train1, train_y, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# In[14]:


clf.fit(train1,train_y)
pred=clf.predict(test1)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# #### smote算法过采样

# In[33]:


# 过抽样处理库SMOTE
from imblearn.over_sampling import SMOTE 

# 建立SMOTE模型对象
model_smote = SMOTE(random_state=2) 
# 输入数据并作过抽样处理
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(train,train_y) 
x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=train.columns)
y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['Churn?']) 


# In[34]:


# 去除特征Eve Calls
train1=x_smote_resampled.drop(['Eve Calls'],axis=1)
clf =GBC(random_state=0)
round(cross_val_score(clf, train1, y_smote_resampled, cv=cv, scoring='precision').mean(),4),round(cross_val_score(clf, train1, y_smote_resampled, cv=cv, scoring='recall').mean(),4)


# In[35]:


clf = GBC(learning_rate=0.1, n_estimators=78,
                            max_depth=5, 
                            min_samples_leaf =1,
                            min_samples_split =52,
                        random_state=0)
clf.fit(train1,y_smote_resampled)
test1=test.drop(['Eve Calls'],axis=1)
pred=clf.predict(test1)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# ####  异常值检测

# #### 基于聚类的小簇划分法及离群点划分法

# In[90]:


df=pd.concat([train,train_y],axis=1)
df=df.reset_index()
df=df.drop('index',axis=1)


# In[91]:


# 首先确定最佳簇的个数
#聚类+手肘法
from sklearn.cluster import KMeans  
  
# 存放每次结果的误差平方和  
SSE = []  
for k in range(1,10):  
    estimator = KMeans(n_clusters=k)  # 构造聚类器  
    estimator.fit(df)  
    SSE.append(estimator.inertia_)  
X = range(1,10)  
plt.xlabel('k')  
plt.ylabel('SSE')  
plt.plot(X,SSE,'o-')  
plt.show()  


# In[92]:


# 聚类+轮廓系数
from sklearn.metrics import silhouette_score
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 存放轮廓系数  
Scores = []  
for k in range(2,10):  
    estimator = KMeans(n_clusters=k)  # 构造聚类器  
    estimator.fit(df)  
    Scores.append(silhouette_score(df,estimator.labels_,metric='euclidean'))  
X = range(2,10)  
plt.xlabel('k')  
plt.ylabel(u'轮廓系数')  
plt.plot(X,Scores,'o-')  
plt.show()  


# In[93]:


k=4
iteration=500
model=KMeans(n_clusters=k,max_iter=iteration)
model.fit(df)
# 丢离远离其他簇的小簇
print pd.Series(model.labels_).value_counts()
pd.DataFrame(model.cluster_centers_)


# In[94]:


# 基于离群点得分检测异常点
threshold=2
df1=pd.concat([df,pd.Series(model.labels_,index=df.index)],axis=1)
df1.columns=list(df.columns)+['cluster']

norm=[]
for i in range(k):
    norm_tmp=df1[[x for x in df1.columns if x not in 'cluster']][df1.cluster==i]-model.cluster_centers_[i]
    # 求出相对距离
    norm_tmp=norm_tmp.apply(np.linalg.norm,axis=1)
    # 求出绝对距离，相对距离/所以样本点到质心的相对距离的中位数
    norm.append(norm_tmp/norm_tmp.median())
norm=pd.concat(norm)


# In[123]:


# 删除异常点
outlier_index=norm[norm>threshold].index
normal_index=[x for x in list(df.index) if x not in list(outlier_index)]
df2=df.loc[normal_index,:]
train_y1=df2.pop("Churn?")


# In[130]:


clf = GBC(learning_rate=0.1, n_estimators=78,
                            max_depth=5, 
                            min_samples_leaf =1,
                            min_samples_split =52,
                        random_state=0)
train1=df2.drop(['Eve Calls'],axis=1)
clf.fit(train1,train_y1)
test1=test.drop(['Eve Calls'],axis=1)
pred=clf.predict(test1)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# #### 特征组合

# In[122]:


clf = GBC(learning_rate=0.1, n_estimators=78,
                            max_depth=5, 
                            min_samples_leaf =1,
                            min_samples_split =52,
                        random_state=0)
train1=train.drop(['Eve Calls'],axis=1)
clf.fit(train1,train_y)
test1=test.drop(['Eve Calls'],axis=1)
pred=clf.predict(test1)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# In[123]:


pd.DataFrame(clf.feature_importances_,index=train1.columns).sort_values(by=0,ascending=False)


# In[124]:


data_tmp=pd.concat([data,y],axis=1)


# In[125]:


# 交叉特征
data_tmp['Charge']=data_tmp['Day Charge']+data_tmp['Eve Charge']+data_tmp['Night Charge']
data_tmp['Mins']=data_tmp['Day Mins']+data_tmp['Eve Mins']+data_tmp['Night Mins']
data_tmp['Calls']=data_tmp['Day Calls']+data_tmp['Eve Calls']+data_tmp['Night Calls']-data_tmp['CustServ Calls']

cols=['Day Charge','Eve Charge','Night Charge','Day Mins','Eve Mins','Night Mins','Day Calls','Eve Calls','Night Calls']
for col in cols:
    del data_tmp[col]


# In[126]:


data_tmp.head()


# In[127]:


#划分训练集和测试集
y=data_tmp.pop('Churn?')
train,test,train_y,test_y=train_test_split(data_tmp,y,test_size=0.3,random_state=0)

clf = GBC(learning_rate=0.1, n_estimators=78,
                            max_depth=5, 
                            min_samples_leaf =1,
                            min_samples_split =52,
                        random_state=0)
clf.fit(train,train_y)
pred=clf.predict(test)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# #### 随机森林调参

# In[23]:


from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import GridSearchCV

# 调节参数n_estimators
param_test1 = {'n_estimators':range(55,90,5)}
gsearch1 = GridSearchCV(estimator = RF(random_state=0), param_grid = param_test1, scoring='recall',iid=False,cv=cv)
gsearch1.fit(train,train_y)
gsearch1.grid_scores_,gsearch1.best_score_,gsearch1.best_params_


# In[26]:


# 调节参数max_depth和min_samples_split
param_test2 = {'max_depth':range(3,8,1), 'min_samples_split':range(2,100,20)}
gsearch2 = GridSearchCV(estimator =RF(n_estimators=gsearch1.best_params_['n_estimators'],random_state=0), param_grid = param_test2, scoring='recall',iid=False, cv=cv)
gsearch2.fit(train,train_y)
gsearch2.grid_scores_,gsearch2.best_score_,gsearch2.best_params_


# In[27]:


# 调节参数min_samples_split和min_samples_leaf
param_test3 = {'min_samples_split':range(2,200,50), 'min_samples_leaf':range(1,100,10)}
gsearch3 = GridSearchCV(estimator = RF(n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], random_state=0), param_grid = param_test3, scoring='recall',iid=False, cv=cv)
gsearch3.fit(train,train_y)
gsearch3.grid_scores_,gsearch3.best_score_,gsearch3.best_params_


# In[32]:


# 调节参数max_features
param_test4 = {'max_features':range(1,train.shape[1]+1,1)}
gsearch4 = GridSearchCV(estimator = RF( n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'], random_state=0), param_grid = param_test4, scoring='recall',iid=False, cv=cv)
gsearch4.fit(train,train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[34]:


# 调参后的Random Forest
clf=RF(n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], 
       min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'],
       max_features=gsearch4.best_params_['max_features'],random_state=0)
clf.fit(train,train_y)
pred=clf.predict(test)
metrics.recall_score(test_y,pred)


# #### 模型融合

# In[162]:


train.shape,train_y.shape,test.shape,test_y.shape


# In[129]:


train=train.reset_index()
train.drop('index',axis=1,inplace=True)

train_y=train_y.reset_index()
train_y.drop('index',axis=1,inplace=True)

test=test.reset_index()
test.drop('index',axis=1,inplace=True)

test_y=test_y.reset_index()
test_y.drop('index',axis=1,inplace=True)


# In[143]:


def ensemble_model(model,train,train_y,test,n_folds=5,random_state=0):
    
    num_train, num_test = train.shape[0], test.shape[0]
    L1_train = np.zeros((num_train,)) 
    L1_test = np.zeros((num_test,))
    L1_test_all = np.zeros((num_test, n_folds))
    KF = KFold(n_splits = n_folds, random_state=random_state)
    
    for i, (train_index, val_index) in enumerate(KF.split(train)):
        x_train, y_train = train[train_index], train_y[train_index]
        x_val, y_val = train[val_index], train_y[val_index]
        model.fit(x_train,y_train)
        L1_train[val_index] = model.predict(x_val)
        L1_test_all[:, i] = model.predict(test)
    L1_test = np.mean(L1_test_all, axis=1)
    
    return L1_train,L1_test


# In[144]:


model=GBC(learning_rate=0.1, n_estimators=78,max_depth=5, min_samples_leaf =1,min_samples_split =52,random_state=0)
gbc_train,gbc_test=ensemble_model(model,train.values,train_y.values,test.values)

model=RF(n_estimators=75,max_depth=7,min_samples_leaf =1, min_samples_split =2, max_features=5,random_state=0)
rf_train,rf_test=ensemble_model(model,train.values,train_y.values,test.values)


# In[146]:


input_train=[gbc_train,rf_train] 
input_test=[gbc_test,rf_test]

stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)


# #### tuning

# In[163]:


train1=stacked_train


# In[166]:


from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(1,42,5)}
gsearch1 = GridSearchCV(estimator = GBC(learning_rate=0.1,random_state=0), 
                   param_grid = param_test1, scoring='recall',iid=False,cv=cv)
gsearch1.fit(train1,train_y)
gsearch1.best_score_,gsearch1.best_params_


# In[169]:


#调节参数max_depth和min_samples_split
param_test2 = {'max_depth':range(3,9,1), 'min_samples_split':range(200,503,50)}
gsearch2 = GridSearchCV(estimator =GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
                                                          random_state=0), 
                    param_grid = param_test2, scoring='recall',iid=False, cv=cv)
gsearch2.fit(train1,train_y)
gsearch2.best_score_,gsearch2.best_params_


# In[172]:


#调节参数min_samples_split和min_samples_leaf
param_test3 = {'min_samples_split':range(200,350,20), 'min_samples_leaf':range(1,10,1)}
gsearch3 = GridSearchCV(estimator = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
                                                          max_depth=gsearch2.best_params_['max_depth'], random_state=0), 
                   param_grid = param_test3, scoring='recall',iid=False, cv=cv)
gsearch3.fit(train1,train_y)
gsearch3.best_score_,gsearch3.best_params_


# In[173]:


clf = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
                            max_depth=gsearch2.best_params_['max_depth'], 
                            min_samples_leaf =gsearch3.best_params_['min_samples_leaf'],
                            min_samples_split =gsearch3.best_params_['min_samples_split'],
                        random_state=0)


# In[174]:


clf.fit(stacked_train, y_train)
pred=clf.predict(stacked_test)
metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)


# In[175]:


clf.get_params

