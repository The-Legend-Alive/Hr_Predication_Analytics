# Importing the libraries
import pandas as pd
df = pd.read_excel (r'New Excell leavers analytics.xlsx')
#print (df)
#print (df.describe(include = 'all'))

# # removing null values to avoid errors 
# df.dropna(inplace = True) 
  
# # percentile list
# perc =[.20, .40, .60, .80]
  
# # list of dtypes to include
# include =['object', 'float', 'int']

# desc = df.describe(percentiles = perc, include = include)

# print(desc) # Personal objectives
df['Business Unit'] = df['Business Unit'].astype('category').cat.codes
df['Starting Level'] = df['Starting Level'].astype('category').cat.codes
df['Current/End Level'] = df['Current/End Level'].astype('category').cat.codes
df['Promotion?'] = df['Promotion?'].astype('category').cat.codes
df['Gender'] = df['Gender'].astype('category').cat.codes
df['Left'] = df['Left'].astype('category')
df['Current/last client'] = df['Current/last client'].astype('category').cat.codes
df['Absenteism?'] = df['Absenteism?'].astype('category').cat.codes
df['Avg Time on project'] = df['Avg Time on project'].astype('category').cat.codes
df['Reasons for leaving'] = df['Reasons for leaving'].astype('category').cat.codes
df['Q6'] = df['Q6'].astype('category').cat.codes
df['q7'] = df['q7'].astype('category').cat.codes
df['#Mission/Projects'] = df['#Mission/Projects'].astype('category').cat.codes
df['Difference in Salary'] = df['Difference in Salary'].astype('category').cat.codes
#df['#Events'] = df['#Events'].astype('category').cat.codes
df['Employee Name'].fillna('',inplace=True)
df['Difference in Salary'].fillna('',inplace=True)
df['#Certifications'].fillna(0,inplace=True)


df['Age'] = pd.to_numeric(df['Age'], downcast='float')
df['Seniority'] = pd.to_numeric(df['Seniority'], downcast='float')

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# sns.lmplot(x='Seniority', y='Age',data=df, hue='Left', 
#            palette='Set1')
# plt.xlabel("Seniority")
# plt.ylabel("Age")
# plt.title("Seniority Vs. Age")
# plt.legend()
# plt.show()

#print(df['#Personal objectives'])

#Modelling the Seniority
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mean = np.mean(df["Seniority"])
std = np.std(df["Seniority"])


# # Function to compute the ECDF
# def ecdf(data):
#     n = len(data)
#     x = np.sort(data)
#     y = np.arange(1, n + 1) / n
#     return x, y


# # trying out a normal distribution
# samples = np.random.normal(mean, std, size=10000)
# x, y = ecdf(df["Seniority"])
# x_theor, y_theor = ecdf(samples)
# _ = plt.plot(x_theor, y_theor)
# _ = plt.plot(x, y, marker='.', linestyle='none')
# plt.xlabel("Seniority")
# plt.ylabel("CDFs")
# plt.show()

#-------------Hypothesis Testing the correlations----------------
#Creating a functon to compute the pearson correlation coeff 
# def pearson_r(x,y):
#     corr_mat = np.corrcoef(x,y)
#     return corr_mat[0,1]

# #Computing the pearson correlation coeff between seniority and age  
# r = pearson_r(df["Seniority"],df["Age"])
# print("Corr Coeff of seniority and age scores : ", r)

# #Computing the pearson correlation coeff between Seniority and Personal objectives  
# r = pearson_r(df["Seniority"],df["# Personal objectives"])
# print("Corr Coeff of seniority and personal objectives scores : ", r)

# #Hypothesis test to see if There is a weak postive correlation between seniority and age  

# r_obs = pearson_r(df["Age"],df["Seniority"])

# perm_replicates = np.empty(120000)
# for i in range(120000):
#     age_permuted = np.random.permutation(df["Age"])
#     perm_replicates[i] = pearson_r(age_permuted,df["Seniority"])
    
# #Computing the p value
# p = np.sum(perm_replicates <= r_obs) / len(perm_replicates)
# print('p-val =', p)

#-------------Machine Learning----------------
 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

for col in df.columns:
    if 'Start Date' in col:
        del df[col]
    if 'End Date' in col:
        del df[col]
    if '#Events' in col:
        del df[col]
    if 'Employee Name' in col:
        del df[col]

df

# The categorical columns are one-hot encoded
X = pd.get_dummies(df.drop('Left', axis=1)).values
y = df['Left'].cat.codes.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train= sc_y.fit_transform(y_train.reshape(-1,1))
# y_train= np.ravel(y_train)

print(y_train)



#-------------Parameter Fine Tuning----------------

#-------------KNeighborsClassifier----------------
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier(n_neighbors=6)
param_grid = {'n_neighbors': np.arange(3, 15)}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_

#-------------Building the machine learning models----------------
knn = KNeighborsClassifier(n_neighbors=8)
logreg = LogisticRegression()
tree = DecisionTreeClassifier()

classification_models = {
    'KNeighboursClassfier': knn,
    'DecisionTreeClassifier': tree
}

regression_models = {
    'LogisticRegression': logreg
}

#------------- Model Scores---------------
for name, model in classification_models.items():
    model.fit(X_train, y_train)
    print('{}\t{}'.format(name, model.score(X_test, y_test)))
    
for name, model in regression_models.items():
    model.fit(X_train, y_train)
    print('{}\t{}'.format(name, model.score(X_test, y_test)))

#------------- ROC Curve and AUC---------------
subplot_count = 1

for name, model in classification_models.items():
    y_pred = model.predict(X_test)
    fpr, tpr, tresholds = roc_curve(y_test, y_pred)
    
    plt.subplot(1, len(classification_models), subplot_count)
    
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr, label=name)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC curve'.format(name))
    
    subplot_count += 1
    
    print('{}\t{}'.format(name, roc_auc_score(y_test, y_pred)))
    
    plt.show()


# subplot_count = 1

# for name, model in regression_models.items():
#     y_pred_prob = model.predict_proba(X_test)[:,1]
#     fpr, tpr, tresholds = roc_curve(y_test, y_pred_prob)
    
#     plt.subplot(1, len(regression_models), subplot_count)
    
#     plt.plot([0,1], [0,1], 'k--')
#     plt.plot(fpr, tpr, label=name)

#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('{} ROC curve'.format(name))
    
#     subplot_count += 1
    
#     print('{} AUC:\t{}'.format(name, roc_auc_score(y_test, y_pred_prob)))
#     plt.show()

# #Scaling: Standardizes features by removing the mean and scaling to unit variance
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# for name, model in classification_models.items():
    
#     steps = [
#         ('scaler', StandardScaler()),
#         (name, model)
#     ]
    
#     pipeline = Pipeline(steps)
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
    
#     print('{}\t{}'.format(name, accuracy_score(y_test, y_pred)))


# for name, model in regression_models.items():
    
#     steps = [
#         ('scaler', StandardScaler()),
#         (name, model)
#     ]
    
#     pipeline = Pipeline(steps)
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
    
#     print('{}\t{}'.format(name, accuracy_score(y_test, y_pred)))

# #Feature Selection
# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier()
# model.fit(X, y)

# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print (pd.DataFrame(model.feature_importances_,
#              index=pd.get_dummies(df.drop('Left', axis=1)).columns,
#              columns=['Importance']))

