# Wasting time on Kaggle competitions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# load the data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Read Data Files
#############################################################################

train_df = pd.read_csv("Shelter/train.csv")
test_df = pd.read_csv("Shelter/test.csv")

new_columns = test_df.columns.values
print new_columns 
new_columns[0] = 'AnimalID'
test_df.columns = new_columns

print test_df.columns.values

all_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)


all_df.info()


# print(test_df.info())
# print(train_df.info())
# convert all of the AgeuponOutcome values into weeks


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Data Extraction functions
#############################################################################

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

def calc_AgeuponOutcome_to_yearGroups(df):
    result={}
    for k in df['_AgeuponOutcome'].unique():
        if k <= 52:
            result[k] = '1 Year'
        elif k >52 and k<=52*3:
            result[k] = '1-3 Years'
        elif k >52*3 and k<=52*5:
            result[k] = '3-5 Years'
        elif k >52*5 and k<=52*7:
            result[k] = '5-7 Years'
        elif k >52*7 and k<=52*10:
            result[k] = '7-10 Years'
        elif k >52*10:
            result[k] = '>10 Years'
                
    df['AgeGroups'] = df['_AgeuponOutcome'].map(result)          
    return df

def extract_field(_df, start, stop):
    return _df['DateTime'].map(lambda dt: int(dt[start:stop]))

def calcYearMonth(df):
    df['Year'] = extract_field(df,0,4)
    df['Month'] = extract_field(df,5,7)
    df['Day'] = extract_field(df,8,10)
    df['Hour'] = extract_field(df,11,13)
    df['Minute'] = extract_field(df,14,16)
    
    return df.drop(['DateTime'], axis = 1)
    
def getColors(df):
    result={}
    color={}
    for k in df['Color'].unique():
        c = k.split("/")
        # print(c, len(c))
        if(len(c) == 1):
            result[k] = "No"
            color[k] = c[0]
        elif(len(c) == 2):
            result[k] = "Yes"
            color[k] = c[0]
    df['MixColor'] = df['Color'].map(result)
    df['firstColor'] = df['Color'].map(color)
    return df

def checkName(df):
    result={}
    for k in df['Name']:
        if type(k) != type(""):
            result[k] = "No"
        else:
            result[k] = "Yes"
    df['hasName'] = df['Name'].map(result)
    return df

def checkBreed(df):
    result={}
    for k in df['Breed']:
        b = k.split(' ')
        if b[len(b)-1] in ["Mix","mix"]:
            result[k] = "Mix"
        else:
            result[k] = "Normal"
    df['mixBreed'] = df['Breed'].map(result)
    return df

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Data Extraction
#############################################################################

# train_df = convert_AgeuponOutcome_to_weeks(train_df)
# test_df = convert_AgeuponOutcome_to_weeks(test_df)
all_df = convert_AgeuponOutcome_to_weeks(all_df)
print("\n\nAgeuponOutcome conversion done.")

# names = pd.concat([test_df['Name'], train_df['Name']])
names = all_df['Name']
values = dict(names.value_counts())


# train_df['_NameFreq'] = train_df['Name'].map(values)
# test_df['_NameFreq'] = test_df['Name'].map(values)
all_df['_NameFreq'] = all_df['Name'].map(values)

# train_df['_NameFreq'] = train_df['_NameFreq'].fillna(0)
# test_df['_NameFreq'] = test_df['_NameFreq'].fillna(0)
all_df['_NameFreq'] = all_df['_NameFreq'].fillna(0)
print("\n\nName Fequency count done.")

# train_df=calc_AgeuponOutcome_to_yearGroups(train_df)
# test_df=calc_AgeuponOutcome_to_yearGroups(test_df)
all_df=calc_AgeuponOutcome_to_yearGroups(all_df)
print("\n\n Age grouping done.")


# train_df=calcYearMonth(train_df)
# test_df=calcYearMonth(test_df)
all_df=calcYearMonth(all_df)
print("\n\n Date Time splitting done.")

# train_df=getColors(train_df)
# test_df=getColors(test_df)
all_df=getColors(all_df)
print("\n\n color categorization done.")

# train_df=checkName(train_df)
# test_df=checkName(test_df)
all_df=checkName(all_df)
print("\n\n Name checking done.")

# train_df=checkBreed(train_df)
# test_df=checkBreed(test_df)
all_df=checkBreed(all_df)
print("\n\n Breed checking done.")


# print train_df.info()
# print train_df
'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plots
#############################################################################
 
AnimalType = train_df['AnimalType'].value_counts() 
AnimalType.plot(kind='bar',color='#34ABD8',rot=0)
plt.show()

NameFrequency = train_df['_NameFreq'].value_counts()
NameFrequency.plot(kind='bar',color='#34ABD8',rot=0)
plt.show()

AnimalType = train_df[['AnimalType','OutcomeType']].groupby(['OutcomeType','AnimalType']).size().unstack()
AnimalType.plot(kind='bar',color=['#34ABD8','#E98F85'],rot=-30)
plt.show()

AnimalType = train_df[['SexuponOutcome','OutcomeType']].groupby(['OutcomeType','SexuponOutcome']).size().unstack()
AnimalType.plot(kind='bar',rot=-30)
plt.show()

AnimalType = train_df[['AgeGroups','OutcomeType']].groupby(['OutcomeType','AgeGroups']).size().unstack()
AnimalType.plot(kind='bar',rot=-30)
plt.show()


AnimalType = train_df[['MixColor','OutcomeType']].groupby(['OutcomeType','MixColor']).size().unstack()
AnimalType.plot(kind='bar',rot=-30, title="outcome by Mixcolor")
plt.show()


AnimalType = train_df[['firstColor','OutcomeType']].groupby(['OutcomeType','firstColor']).size().unstack()
AnimalType.plot(kind='bar',rot=-30, title="outcome by First color")
plt.show()

AnimalType = train_df[['hasName','OutcomeType']].groupby(['OutcomeType','hasName']).size().unstack()
AnimalType.plot(kind='bar',rot=-30, title="outcome by Name")
plt.show()


AnimalType = train_df[['mixBreed','OutcomeType']].groupby(['OutcomeType','mixBreed']).size().unstack()
AnimalType.plot(kind='bar',rot=-30, title="outcome by Mix Breed")
plt.show()

OutcomeByYear = train_df['Year'].value_counts()
# print OutcomeByYear.values
# print type(OutcomeByYear)
keys = OutcomeByYear.keys()
# print keys
OutcomeByYear.sort_index(0)
# print dir(OutcomeByMonth)
# print OutcomeByYear

OutcomeByYear.plot(kind='bar',rot=0, title="outcome by year")
plt.show()



OutcomeByMonth = train_df['Month'].value_counts()
print OutcomeByMonth.values
# print dir(OutcomeByMonth)
print OutcomeByMonth

OutcomeByMonth.plot(kind='bar',rot=0, title="Outcome By Month")
# plt.bar(indexes, values,)
plt.show()



# feature = 'AgeGroups'

# feature_out_values_dog = np.array(train_df.loc[train_df['AnimalType'] == 'Dog',feature])
# # outcome_dog = np.array(train_df.loc[train_df['AnimalType'] == 'Dog',])

# AnimalType = feature_out_values_dog[['_AgeGroups','OutcomeType']].groupby(['OutcomeType','_AgeGroups']).size().unstack()
# AnimalType.plot(kind='bar',rot=-30)
# plt.show()

################################################################################
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Numeric Conversion of Data
#############################################################################


# print(train_df)
# print(train_df.info())

def convert_to_numeric(df):
    for col in ['Name', 'AnimalType', 'SexuponOutcome',
                'Breed', 'Color', 'AgeGroups', 'MixColor', 'firstColor', 'hasName', 'mixBreed',
                'OutcomeType']:
        if col in df.columns:
            _col = "_%s" % (col)
            values = df[col].unique()
            _values = dict(zip(values, range(len(values))))
            df[_col] = df[col].map(_values).astype(int)
            df = df.drop(col, axis = 1)
    return df

# train_df = convert_to_numeric(train_df)
# test_df = convert_to_numeric(test_df)
all_df = convert_to_numeric(all_df)

print(all_df.info())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Applying algorithms on data
#############################################################################


print("\nNumerical conversion of features done.")

features = ['_Name', '_NameFreq', '_hasName',
            '_AnimalType', '_SexuponOutcome', '_AgeuponOutcome', '_AgeGroups', 
            '_Breed', '_mixBreed', '_Color', '_MixColor', '_firstColor',
            'Year', 'Month', 'Day', 'Hour', 'Minute']

# features =  ['_hasName',
#             '_AnimalType', '_SexuponOutcome', '_AgeuponOutcome', '_AgeGroups', 
#             '_Breed', '_mixBreed', '_Color', '_MixColor', '_firstColor',
#              'Month', 'Day']

# train_df = train_df.reindex(columns = ['_hasName',
#                                        '_AnimalType', '_SexuponOutcome', '_AgeuponOutcome', 
#                                        '_mixBreed', '_Color', '_MixColor',
#                                        'Year', 'Month',
#                                        '_OutcomeType'])

# train_x = train_df[features]
# test_x = test_df[features]
# train_y =train_df['_OutcomeType']

train_x = all_df[:26729]
test_x = all_df[26729:]
train_x = train_x.reindex(columns = features)
test_x = test_x.reindex(columns = features)


train_y =all_df['_OutcomeType'][:26729]


print "Train info"
print train_x.info()
print "Train output"
print train_y

print "test info"
print test_x.info()
print test_x
# print(train_df.info())
# print(train_df)

cut = int(len(train_x) * 0.8)
_validation_x = train_x[cut:]
_validation_y = train_y[cut:]
_train_x = train_x[:cut]
_train_y = train_y[:cut]

print "Train data : ",len(_train_x)
print "Test data : ",len(_validation_x)

import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

print "Starting RandomForestClassifier"
A1 = RandomForestClassifier(n_estimators=1000, max_features= 10, max_depth= 10)
A2 = ExtraTreesClassifier(n_estimators=1000, max_features= 10, max_depth= 15)

A3 = RandomForestClassifier(n_estimators=1000, max_features= 5, max_depth= 10)
A4 = ExtraTreesClassifier(n_estimators=1000, max_features= 5, max_depth= 15)
A5 = RandomForestClassifier(n_estimators=1000, max_features= 4, max_depth= 10)
A6 = ExtraTreesClassifier(n_estimators=1000, max_features= 4, max_depth= 15)
A7 = RandomForestClassifier(n_estimators=1000, max_features= 3, max_depth= 10)
A8 = ExtraTreesClassifier(n_estimators=1000, max_features= 3, max_depth= 15)


A9 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6),
                        n_estimators = 1000,
                        learning_rate = 0.1)
A10 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5,
     max_depth=5, random_state=0)

#submission
#A3 = RandomForestClassifier(n_estimators=1000, max_features= 10, max_depth= 10)

# y_pred = A1.fit(_train_df, _train_df['_OutcomeType']).predict(_validation_df)
# y_true = _validation_df['_OutcomeType']

# print y_pred
# print y_true

# print "Accuracy Random Forest : ", accuracy_score(y_true, y_pred)

# print _train_df.values[:,0:-1]
# print _train_df.values[:,-1]

# classifiers = [c.fit(train_df.values[:,1:-1],
#                      train_df.values[:,-1].astype(int)) \
#                for c in [A1,A2,A3]]


classifiers = [c.fit(_train_x, _train_y.astype(int)) \
               for c in [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]]


# clf = A3.fit(train_x,train_y)

results = [c.predict_proba(_validation_x) \
           for c in classifiers]


# results = clf.predict_proba(test_x)
          
# predicted = np.array(A3.predict_proba(test_df))

# result=pd.DataFrame(results, columns=['Return_to_owner','Euthanasia','Adoption','Transfer','Died'])
# result.to_csv("RandomForest.csv", index = True, index_label = 'ID')

# print([log_loss(_validation_df.values[:,-1].astype(int), r) for r in results])

print([log_loss(_validation_y.astype(int), r) for r in results])