# Wasting time on Kaggle competitions

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# load the data

train_df = pd.read_csv("Shelter/train.csv")
test_df = pd.read_csv("Shelter/test.csv")

# print(test_df.info())
# print(train_df.info())

# convert all of the AgeuponOutcome values into weeks

def convert_AgeuponOutcome_to_weeks(df):
    result = {}
    for k in df['AgeuponOutcome'].unique():
        if type(k) != type(""):
            result[k] = -1
        else:
            v1, v2 = k.split()
            if v2 in ["year", "years"]:
                result[k] = int(v1) * 52
            elif v2 in ["month", "months"]:
                result[k] = int(v1) * 4.5
            elif v2 in ["week", "weeks"]:
                result[k] = int(v1)
            elif v2 in ["day", "days"]:
                result[k] = int(v1) / 7
                
    df['_AgeuponOutcome'] = df['AgeuponOutcome'].map(result).astype(float)
    df = df.drop('AgeuponOutcome', axis = 1)
                
    return df
    
    
# def calc_AgeuponOutcome_to_yearGroups(df):
#     result={}
#     for k in df['_AgeuponOutcome'].unique():
#         if k <= 52:
#             result[k] = '1 Year'
#         elif k >52 and k<=52*3:
#             result[k] = '1-3 Years'
#         elif k >52*3 and k<=52*5:
#             result[k] = '3-5 Years'
#         elif k >52*5 and k<=52*7:
#             result[k] = '5-7 Years'
#         elif k >52*7 and k<=52*10:
#             result[k] = '7-10 Years'
#         elif k >52*10:
#             result[k] = '>10 Years'
                
#     df['_AgeGroups'] = df['_AgeuponOutcome'].map(result)
#     df = df.drop('_AgeuponOutcome', axis = 1)            
#     return df


#############################################################################
train_df = convert_AgeuponOutcome_to_weeks(train_df)
test_df = convert_AgeuponOutcome_to_weeks(test_df)
print("\n\nAgeuponOutcome conversion done.")

# train_df=calc_AgeuponOutcome_to_yearGroups(train_df)
# test_df=calc_AgeuponOutcome_to_yearGroups(test_df)

names = pd.concat([test_df['Name'], train_df['Name']])
values = dict(names.value_counts())

train_df['_NameFreq'] = train_df['Name'].map(values)
test_df['_NameFreq'] = test_df['Name'].map(values)

train_df['_NameFreq'] = train_df['_NameFreq'].fillna(99999)
test_df['_NameFreq'] = test_df['_NameFreq'].fillna(99999)

print(train_df)

print(test_df.info())
print(train_df.info())

print("\nName frequency count done.")

# convert all of the remaining features to numeric values

def convert_to_numeric(df):
    for col in ['Name', 'AnimalType', 'SexuponOutcome',
                'Breed', 'Color', 'OutcomeType']:
        if col in df.columns:
            _col = "_%s" % (col)
            values = df[col].unique()
            _values = dict(zip(values, range(len(values))))
            df[_col] = df[col].map(_values).astype(int)
            df = df.drop(col, axis = 1)
    return df

train_df = convert_to_numeric(train_df)
test_df = convert_to_numeric(test_df)
print(train_df)


print("\nNumerical conversion of features done.")

# fix the DateTime column

def fix_date_time(df):
    def extract_field(_df, start, stop):
        return _df['DateTime'].map(lambda dt: int(dt[start:stop]))
    df['Year'] = extract_field(df,0,4)
    df['Month'] = extract_field(df,5,7)
    df['Day'] = extract_field(df,8,10)
    df['Hour'] = extract_field(df,11,13)
    df['Minute'] = extract_field(df,14,16)
    
    return df.drop(['DateTime'], axis = 1)

train_df = fix_date_time(train_df)
test_df = fix_date_time(test_df)

print("DateTime column split into parts done.")

# re-index train_df so that ID is first and Target (_OutcomeType) is last

train_df = train_df.reindex(columns = ['_Name', '_NameFreq',
                                       '_AnimalType', '_SexuponOutcome','_AgeGroups',
                                       '_Breed', '_Color',
                                       'Year', 'Month', 'Day', 'Hour', 'Minute',
                                       '_OutcomeType'])
                                                                              

                                       
print(train_df.info())
print(test_df.info())

# split the data into a training set (80%) and a validation set (20%)

cut = int(len(train_df) * 0.8)
_validation_df = train_df[cut:]
_train_df = train_df[:cut]

print(len(_train_df))
print(len(_validation_df))

'''
# build a classifier with scikit-learn

import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

A1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2),
                        n_estimators = 100,
                        learning_rate = 0.1)

classifiers = [c.fit(_train_df.values[:,1:-1],
                     _train_df.values[:,-1].astype(int)) \
               for c in [A1]]
results = [c.predict_proba(_validation_df.values[:,1:-1]) \
           for c in classifiers]

print("DecisionTreeClassifier: ",results[0])


# calculate the log loss of result
from sklearn.metrics import log_loss
print([log_loss(_validation_df.values[:,-1].astype(int), r) for r in results])


# re-build the selected classifier on the entire training set

ab = classifiers[0].fit(train_df.values[:,1:-1],
                        train_df.values[:,-1].astype(int))

# and use the classifier on test_df

ab_result = ab.predict_proba(test_df.values[:,1:])
ab_sub_df = pd.DataFrame(ab_result, columns=['Adoption', 'Died', 'Euthanasia',
                                             'Return_to_owner', 'Transfer'])
ab_sub_df.insert(0, 'ID', test_df.values[:,0].astype(int))

print(ab_sub_df)

# write to submission files

ab_sub_df.to_csv("submission.csv", index = False)

print("D'one.")
'''
'''
import sklearn
 
training, validation = train_test_split(train_data, train_size=.60)
model = BernoulliNB()
model.fit(train_df, training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted) 
'''


# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(train_df, train_df['_OutcomeType'])
# predicted = np.array(gnb.predict_proba(test_df))
# result=pd.DataFrame(predicted, columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
# result.to_csv("naiveBayes.csv", index = True, index_label = 'ID')
# #y_pred = gnb.fit(train_df, train_df['_OutcomeType']).predict(test_df)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(_train_df, _train_df['_OutcomeType'])
y_pred = gnb.fit(_train_df, _train_df['_OutcomeType']).predict(_validation_df)
# predicted = np.array(gnb.predict_proba(_validation_df))
# result=pd.DataFrame(predicted, columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])



y_true = _validation_df['_OutcomeType']
# # print _validation_df['_OutcomeType']
# # print y_pred
print("Naive bayes")
print precision_recall_fscore_support(y_true, y_pred, average='macro')
print precision_recall_fscore_support(y_true, y_pred, average='micro')
print precision_recall_fscore_support(y_true, y_pred, average='weighted')


'''
print(y_pred)

prediction = []
for i in range(len(y_pred)):
    testId = test_df['ID'][i].astype(int)
    if(y_pred[i] == 0):
        prediction.append([testId,1,0,0,0,0])
    elif (y_pred[i] == 1):
        prediction.append([testId,0,1,0,0,0])
    elif (y_pred[i] == 2):
        prediction.append([testId,0,0,1,0,0])
    elif (y_pred[i] == 3):
        prediction.append([testId,0,0,0,1,0])
    elif (y_pred[i] == 4):
        prediction.append([testId,1,0,0,0,1])
    
#print(prediction)
prediction.to_csv("naiveBayes.csv");
'''

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
# clf.fit(train_df, train_df['_OutcomeType'])
y_pred = clf.fit(_train_df, _train_df['_OutcomeType']).predict(_validation_df)

# prediction = clf.fit(train_df, train_df['_OutcomeType']).predict(test_df)
# predicted = np.array(clf.predict_proba(test_df))

# result=pd.DataFrame(predicted, columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
# result.to_csv("randomForest.csv", index = True, index_label = 'ID')


print("Random Forest")
print precision_recall_fscore_support(y_true, y_pred, average='macro')
print precision_recall_fscore_support(y_true, y_pred, average='micro')
print precision_recall_fscore_support(y_true, y_pred, average='weighted')




# from sklearn import tree
# treeClf = tree.DecisionTreeClassifier()
# y_pred = treeClf.fit(_train_df, _train_df['_OutcomeType']).predict(_validation_df)
# print("decisionTree")
# print precision_recall_fscore_support(y_true, y_pred, average='macro')
# print precision_recall_fscore_support(y_true, y_pred, average='micro')
# print precision_recall_fscore_support(y_true, y_pred, average='weighted')

'''
print train_df.info()
print("hi")

train_df = train_df.drop('Year', axis = 1)                                       
train_df = train_df.drop('Month', axis = 1)
train_df = train_df.drop('Day', axis = 1)
train_df = train_df.drop('Hour', axis = 1)
train_df = train_df.drop('Minute', axis = 1)
# train_df = train_df.drop('_AgeuponOutcome', axis = 1)

test_df = test_df.drop('Year', axis = 1)                                       
test_df = test_df.drop('Month', axis = 1)
test_df = test_df.drop('Day', axis = 1)
test_df = test_df.drop('Hour', axis = 1)
test_df = test_df.drop('Minute', axis = 1)
# test_df = test_df.drop('_AgeuponOutcome', axis = 1)

print train_df.info()
print("ih")
clf = RandomForestClassifier(n_estimators=10)
clf.fit(train_df, train_df['_OutcomeType'])
# y_pred = clf.fit(_train_df, _train_df['_OutcomeType']).predict(_validation_df)

predicted = np.array(clf.predict_proba(test_df))

result=pd.DataFrame(predicted, columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
result.to_csv("randomForest.csv", index = True, index_label = 'ID')
print("random created")
'''
# print("Random Forest")
# print precision_recall_fscore_support(y_true, y_pred, average='macro')
# print precision_recall_fscore_support(y_true, y_pred, average='micro')
# print precision_recall_fscore_support(y_true, y_pred, average='weighted')
