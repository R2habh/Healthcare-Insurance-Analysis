#!/usr/bin/env python
# coding: utf-8

# ## Healthcare Insurance Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


hospital = pd.read_csv("Hospitalisation details.csv")
medical = pd.read_csv("Medical Examinations.csv")
customer = pd.read_excel("Names.xlsx")


# In[3]:


hospital.head()


# In[4]:


hospital.shape


# In[5]:


hospital.describe()


# In[6]:


hospital.info()


# In[7]:


medical.head()


# In[8]:


medical.shape


# In[9]:


medical.describe()


# In[10]:


medical.info()


# In[11]:


customer.head()


# In[12]:


customer.shape


# In[13]:


customer.describe()


# In[14]:


customer.info()


# ### Collate the files so that all the information is in one place

# In[15]:


df = pd.merge(pd.merge(hospital,medical,on='Customer ID'),customer,on='Customer ID')


# In[16]:


df


# In[17]:


df.shape


# ### Check for missing values in the dataset

# In[18]:


df.isna().sum().sum()


# In[ ]:





# #### Find the percentage of rows that have trivial value (for example, ?), and delete such rows if they do not contain significant information

# In[20]:


trivial = df[df.eq("?").any(axis=1)]


# In[21]:


trivial.shape


# In[22]:


round(trivial.shape[0]/df.shape[0]*100, 2)


# In[24]:


df.drop(df[df.eq("?").any(axis=1)].index, axis=0, inplace=True)


# In[25]:


df.shape


# #### Use the necessary transformation methods to deal with the nominal and ordinal categorical variables in the dataset

# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


l = LabelEncoder()


# In[28]:


df['smoker'] = l.fit_transform(df['smoker'])


# In[29]:


df['Any Transplants'] = l.fit_transform(df['Any Transplants'])


# In[30]:


df['Heart Issues'] = l.fit_transform(df['Heart Issues'])


# In[31]:


df['Cancer history'] = l.fit_transform(df['Cancer history'])


# In[32]:


df['Cancer history'].value_counts()


# In[33]:


df['Heart Issues'].value_counts()


# In[34]:


df['Any Transplants'].value_counts()


# In[35]:


df['smoker'].value_counts()


# In[36]:


def fun(val):
    return int(val.replace("tier", "").replace(" ", "").replace("-", ""))


# In[37]:


df['Hospital tier'] = df['Hospital tier'].map(fun)


# In[38]:


df['City tier'] = df['City tier'].map(fun)


# In[39]:


df


# 

# #### The dataset has State ID, which has around 16 states. All states are not represented in equal proportions in the data. Creating dummy variables for all regions may also result in too many insignificant predictors. Nevertheless, only R1011, R1012, and R1013 are worth investigating further. Create a suitable strategy to create dummy variables with these restraints.

# In[40]:


df['State ID'].value_counts()


# In[41]:


Dummies = pd.get_dummies(df["State ID"], prefix= "State_ID")
Dummies


# In[42]:


Dummy = Dummies[['State_ID_R1011','State_ID_R1012', 'State_ID_R1013']]
Dummy


# In[43]:


df = pd.concat([df,Dummy],axis=1)


# In[44]:


df


# 

# #### The variable NumberOfMajorSurgeries also appears to have string values. Apply a suitable method to clean up this variable.

# In[45]:


df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].replace('No major surgery',0)


# In[46]:


df['NumberOfMajorSurgeries'].value_counts()


# 

# #### Age appears to be a significant factor in this analysis. Calculate the patients' ages based on their dates of birth.

# In[47]:


df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year
df['year']


# In[48]:


df['month'] = pd.to_datetime(df['month'], format='%b').dt.month
df['month']


# In[49]:


df['DateInt'] = df['year'].astype(str) + df['month'].astype(str).str.zfill(2) + df['date'].astype(str).str.zfill(2)


# In[50]:


df['DOB'] = pd.to_datetime(df.DateInt, format='%Y%m%d')


# In[51]:


df.drop(['DateInt'], inplace=True, axis=1)


# In[52]:


import datetime as dt
current_date = dt.datetime.now()


# In[53]:


df['Age'] = (((current_date-df.DOB).dt.days)/365).astype(int)


# In[54]:


df


# 

# #### The gender of the patient may be an important factor in determining the cost of hospitalization. The salutations in a beneficiary's name can be used to determine their gender. Make a new field for the beneficiary's gender.

# In[55]:


def gen(x):
    if 'Ms.' in x:
        return 0
    else:
        return 1


# In[56]:


df['Gender'] = df['name'].map(gen)


# In[57]:


df['Gender']


# #### You should also visualize the distribution of costs using a histogram, box and whisker plot, and swarm plot.

# In[58]:


# Histogram 
sns.histplot(df['charges'])


# In[59]:


# box and whisker plot
sns.boxplot(df['charges'])


# In[61]:


import warnings
warnings.filterwarnings("ignore")


# In[62]:


# Swarm Plot
plt.figure(figsize=(30,10))
sns.swarmplot(x='year', y='charges', hue="Gender", data=df)


# 

# #### State how the distribution is different across gender and tiers of hospitals

# In[63]:


sns.countplot(data = df,x = 'Hospital tier', hue = 'Gender')


# 

# #### Create a radar chart to showcase the median hospitalization cost for each tier of hospitals

# In[64]:


df[df['Hospital tier']==1].charges.median()


# In[65]:


df[df['Hospital tier']==2].charges.median()


# In[66]:


df[df['Hospital tier']==3].charges.median()


# In[67]:


import plotly.express as px

df1 = pd.DataFrame(dict(
    r=[32097.434999999998,7168.76,10676.83],
    theta=['Tier 1','Tier 2','Tier 3']))
fig = px.line_polar(df1, r='r', theta='theta', line_close=True)
fig.show()


# #### Create a frequency table and a stacked bar chart to visualize the count of people in the different tiers of cities and hospitals

# In[68]:


city_freq = df["City tier"].value_counts().rename_axis('City&hospital_tier').reset_index(name='city_counts')


# In[69]:


hospital_freq = df["Hospital tier"].value_counts().rename_axis('City&hospital_tier').reset_index(name='hospital_counts')


# In[70]:


freq_table = pd.merge(city_freq, hospital_freq, on = 'City&hospital_tier')


# In[71]:


freq_table


# In[72]:


x = freq_table['City&hospital_tier']
y1 = freq_table['city_counts']
y2 = freq_table['hospital_counts']
 
# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.show()


# #### Test the following null hypotheses:

# #### The average hospitalization costs for the three types of hospitals are not significantly different

# In[73]:


from scipy.stats import ttest_1samp


# In[74]:


from scipy.stats import friedmanchisquare
data1 = [32097.43]
data2 = [7168.76]
data3 = [10676.83]
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# #### The average hospitalization costs for the three types of cities are not significantly different

# In[75]:


print("median cost of tier 1 city:", df[df["City tier"]==1].charges.median())
print("median cost of tier 2 city:", df[df["City tier"]==2].charges.median())
print("median cost of tier 3 city:", df[df["City tier"]==3].charges.median())


# In[76]:


data1 = [10027.15]
data2 = [8968.33]
data3 = [9880.07]
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# #### The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers

# In[77]:


print("median cost of smoker:", df[df["smoker"]==1].charges.median())
print("median cost of non smoker:", df[df["smoker"]==0].charges.median())


# In[78]:


from scipy.stats import kruskal
data1 = [34125.475]
data2 = [7537.16]
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# #### Smoking and heart issues are independent

# In[79]:


from scipy.stats import chi2_contingency
table = [[df["Heart Issues"].value_counts()],[df["smoker"].value_counts()]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')


# 

# # Machine Learning

# #### Examine the correlation between predictors to identify highly correlated predictors. Use a heatmap to visualize this.

# In[80]:


df.drop(["Customer ID","State ID",'name', 'year', 'month', 'date', 'DOB'], inplace=True, axis=1)


# In[81]:


df.head()


# Dropped those columns because they are not usable to model building

# In[82]:


df.shape


# In[83]:


df.head()


# In[84]:


correlation = df.corr()


# In[85]:


correlation


# In[86]:


plt.figure(figsize=(15,10))
sns.heatmap(correlation, annot=True, linewidth=.5, cmap="crest")
plt.show()


# #### 2. Develop and evaluate the final model using regression with a stochastic gradient descent optimizer. Also, ensure that you apply all the following suggestions:
# • Perform the stratified 5-fold cross-validation technique for model building and validation • Use standardization and hyperparameter tuning effectively • Use sklearn-pipelines • Use appropriate regularization techniques to address the bias-variance trade-off
# 
# 

# #### a. Create five folds in the data, and introduce a variable to identify the folds

# #### b. For each fold, run a for loop and ensure that 80 percent of the data is used to train the model and the remaining 20 percent is used to validate it in each iteration

# #### c. Develop five distinct models and five distinct validation scores (root mean squared error values)

# #### d. Determine the variable importance scores, and identify the redundant variables

# In[87]:


# lets first seperate the input and output data.
x = df.drop(["charges"], axis=1)
y = df[['charges']]


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


x_train, x_test, y_train, y_test  = train_test_split(x,y, test_size=.20, random_state=10)


# In[90]:


from sklearn.preprocessing import StandardScaler


# In[91]:


sc = StandardScaler()


# In[92]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[93]:


# Stochastic gradient descent optimizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV


# In[94]:


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2,0.3,0.4,0.5,
                   0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,
                   9.0,10.0,20,50,100,500,1000],
         'penalty': ['l2', 'l1', 'elasticnet']}

sgd = SGDRegressor()

# Cross Validation 
folds = 5
model_cv = GridSearchCV(estimator = sgd,
                       param_grid = params,
                       scoring = 'neg_mean_absolute_error',
                       cv = folds,
                       return_train_score = True,
                       verbose = 1)
model_cv.fit(x_train,y_train)


# In[95]:


model_cv.best_params_


# In[96]:


sgd = SGDRegressor(alpha= 100, penalty= 'l1')


# In[97]:


sgd.fit(x_train, y_train)


# In[98]:


sgd.score(x_test, y_test)


# In[99]:


y_pred = sgd.predict(x_test)


# In[100]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[101]:


sgd_mae = mean_absolute_error(y_test, y_pred)
sgd_mse = mean_squared_error(y_test, y_pred)
sgd_rmse = sgd_mse*(1/2.0)


# In[102]:


print("MAE:", sgd_mae)
print("MSE:", sgd_mse)
print("RMSE:", sgd_rmse)


# In[103]:


importance = sgd.coef_


# In[104]:


pd.DataFrame(importance, index = x.columns, columns=['Inportance Score'])


# 

# #### Use random forest and extreme gradient boosting for cost prediction, share your crossvalidation results, and calculate the variable importance scores

# #### Random Forest Algorithm

# In[105]:


from sklearn.ensemble import RandomForestRegressor


# In[106]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# In[107]:


rf.fit(x_train, y_train)


# In[108]:


score = rf.score(x_test,y_test)
score


# In[109]:


y_pred = rf.predict(x_test)


# In[110]:


rf_mae = mean_absolute_error(y_test, y_pred)


# In[111]:


rf_mae


# ### Extreme gradient boosting

# In[112]:


from sklearn.ensemble import GradientBoostingRegressor


# In[113]:


gbr = GradientBoostingRegressor(n_estimators = 1000, random_state = 42)


# In[114]:


gbr.fit(x_train, y_train)


# In[115]:


score = gbr.score(x_test,y_test)
score


# In[116]:


y_pred = gbr.predict(x_test)


# In[117]:


gbr_mae = mean_absolute_error(y_test, y_pred)
gbr_mae


# #### 1. Case scenario: Estimate the cost of hospitalization for Christopher, Ms. Jayna (her date of birth is 12/28/1988, height is 170 cm, and weight is 85 kgs). She lives in a tier-1 city and her state’s State ID is R1011. She lives with her partner and two children. She was found to be nondiabetic (HbA1c = 5.8). She smokes but is otherwise healthy. She has had no transplants or major surgeries. Her father died of lung cancer. Hospitalization costs will be estimated using tier-1 hospitals.

# In[118]:


date = str(19881228)
date1 = pd.to_datetime(date, format = "%Y%m%d")


# In[119]:


current_date = dt.datetime.now()
current_date


# In[120]:


age = (current_date - date1)
age


# In[121]:


age = int(12421/365)
age


# So the age of Christopher, Ms. Jayna is 34

# In[122]:


height_m = 170/100
height_sq = height_m*height_m
BMI = 85/height_sq
np.round(BMI,2)


# BMI is 29.41

# In[123]:


df.columns


# In[124]:


df.drop(['charges'],inplace=True,axis=1)


# In[125]:


list = [[2,1,1,29.41,5.8,0,0,0,0,1,1,0,0,34,0]]


# In[126]:


df = pd.DataFrame(list, columns = ['children','Hospital tier', 'City tier', 'BMI', 'HBA1C','Heart Issues', 'Any Transplants', 
                              'Cancer history','NumberOfMajorSurgeries', 'smoker', 'State_ID_R1011', 'State_ID_R1012',
                              'State_ID_R1013', 'age', 'gender'] )
df


# #### Find the predicted hospitalization cost using all five models. The predicted value should be the mean of the five models' predicted values'.

# In[127]:


Hospitalization_cost = []


# In[128]:


# Now lets predict the hospitalization cost through SGDRegressor
Cost1 = sgd.predict(df)
Hospitalization_cost.append(Cost1)


# In[129]:


# Now lets predict the hospitalization cost through Random Forest
Cost2 = rf.predict(df)
Hospitalization_cost.append(Cost2)


# In[130]:


# Now lets predict the hospitalization cost throug Extreme gradient Booster
Cost3 = gbr.predict(df)
Hospitalization_cost.append(Cost3)


# In[131]:


Hospitalization_cost


# In[132]:


avg_cost = np.mean(Hospitalization_cost)
avg_cost


# The average predicted hospitalization cost is 104814.30
