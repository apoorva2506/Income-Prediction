import math
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df_train = pd.read_csv("census_train.csv", names=['idnum', 'age', 'workerclass', 'interestincome', 'traveltimetowork',
'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'educationalattain', 'sex',
'workarrivaltime', 'hoursworkperweek', 'ancestry', 'degreefield', 'industryworkedin', 'wages'])

df_test = pd.read_csv("census_test.csv",names=['idnum', 'age', 'workerclass', 'interestincome', 'traveltimetowork',
'vehicleoccupancy', 'meansoftransport', 'marital', 'schoolenrollment', 'educationalattain', 'sex',
'workarrivaltime', 'hoursworkperweek', 'ancestry', 'degreefield', 'industryworkedin'])


def dataCleaning(dataToBeCleaned):
    dataToBeCleaned['age'] = dataToBeCleaned['age'].replace('?', dataToBeCleaned['age'].mean())
    # dataToBeCleaned = dataToBeCleaned[dataToBeCleaned.age > 16]
    dataToBeCleaned['workerclass'] = dataToBeCleaned['workerclass'].replace('?', 0)
    dataToBeCleaned['traveltimetowork'] = dataToBeCleaned['traveltimetowork'].replace('?', 0)
    dataToBeCleaned['meansoftransport'] = dataToBeCleaned['meansoftransport'].replace('?', 0)
    dataToBeCleaned['vehicleoccupancy'] = dataToBeCleaned['vehicleoccupancy'].replace('?', 0)
    dataToBeCleaned['workarrivaltime'] = dataToBeCleaned['workarrivaltime'].replace('?', 0)
    dataToBeCleaned['hoursworkperweek'] = dataToBeCleaned['hoursworkperweek'].replace('?', 0)
    dataToBeCleaned['degreefield'] = dataToBeCleaned['degreefield'].replace('?', 0)
    dataToBeCleaned['industryworkedin'] = dataToBeCleaned['industryworkedin'].replace('?', 0)

    # drop columns which are not required

    dataToBeCleaned.drop(columns=['ancestry'])

    dataToBeCleaned['workerclass'] = dataToBeCleaned['workerclass'].replace(['1', '2'], 1)
    dataToBeCleaned['workerclass'] = dataToBeCleaned['workerclass'].replace(['3', '4', '5'], 2)
    dataToBeCleaned['workerclass'] = dataToBeCleaned['workerclass'].replace(['6', '7'], 3)
    dataToBeCleaned['workerclass'] = dataToBeCleaned['workerclass'].replace(['8', '9'], 4)
    dataToBeCleaned['educationalattain'] = dataToBeCleaned['educationalattain'].replace([15, 16, 17, 18, 19], 1)

    # merging occupancy transpot, vehicle occupancy and means

    dataToBeCleaned['occupancytransport'] = 0
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '1') & (dataToBeCleaned['meansoftransport'] == '1')] = 1
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '2') & (dataToBeCleaned['meansoftransport'] == '1')] = 2
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '3') & (dataToBeCleaned['meansoftransport'] == '1')] = 3
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '4') & (dataToBeCleaned['meansoftransport'] == '1')] = 4
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '5') & (dataToBeCleaned['meansoftransport'] == '1')] = 5
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '6') & (dataToBeCleaned['meansoftransport'] == '1')] = 6
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '7') & (dataToBeCleaned['meansoftransport'] == '1')] = 7
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '8') & (dataToBeCleaned['meansoftransport'] == '1')] = 8
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '9') & (dataToBeCleaned['meansoftransport'] == '1')] = 9
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '10') & (dataToBeCleaned['meansoftransport'] == '1')] = 10
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '2')] = 11
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '3')] = 12
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '4')] = 13
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '5')] = 14
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '6')] = 15
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '7')] = 16
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '8')] = 17
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '9')] = 18
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '10')] = 19
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '11')] = 20
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '12')] = 21
    dataToBeCleaned['occupancytransport'][(dataToBeCleaned['vehicleoccupancy'] == '?') & (dataToBeCleaned['meansoftransport'] == '?')] = 22

    dataToBeCleaned['workarrivaltime'] = pd.to_numeric(dataToBeCleaned['workarrivaltime'])
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] > 0) & (dataToBeCleaned['workarrivaltime'] < 87)] = 1
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 87) & (dataToBeCleaned['workarrivaltime'] < 94)] = 2
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 94) & (dataToBeCleaned['workarrivaltime'] < 100)] = 3
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 100) & (dataToBeCleaned['workarrivaltime'] < 106)] = 4
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 106) & (dataToBeCleaned['workarrivaltime'] < 112)] = 5
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 112) & (dataToBeCleaned['workarrivaltime'] < 118)] = 6
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 118) & (dataToBeCleaned['workarrivaltime'] < 142)] = 7
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 142) & (dataToBeCleaned['workarrivaltime'] < 202)] = 8
    dataToBeCleaned['workarrivaltime'][(dataToBeCleaned['workarrivaltime'] >= 202) & (dataToBeCleaned['workarrivaltime'] <= 285)] = 9

    # changing all data to numeric

    dataToBeCleaned['traveltimetowork'] = pd.to_numeric(dataToBeCleaned['traveltimetowork'])
    dataToBeCleaned['hoursworkperweek'] = pd.to_numeric(dataToBeCleaned['hoursworkperweek'])
    dataToBeCleaned['industryworkedin'] = pd.to_numeric(dataToBeCleaned['industryworkedin'])

    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 170) & (dataToBeCleaned['industryworkedin'] < 370)] = 1
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 370) & (dataToBeCleaned['industryworkedin'] < 570)] = 2
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 570) & (dataToBeCleaned['industryworkedin'] < 770)] = 3
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 770) & (dataToBeCleaned['industryworkedin'] < 1070)] = 4
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 1070) & (dataToBeCleaned['industryworkedin'] < 4070)] = 5
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 4070) & (dataToBeCleaned['industryworkedin'] < 4670)] = 6
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 4670) & (dataToBeCleaned['industryworkedin'] < 6070)] = 7
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 6070) & (dataToBeCleaned['industryworkedin'] < 6470)] = 8
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 6470) & (dataToBeCleaned['industryworkedin'] < 6870)] = 9
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 6870) & (dataToBeCleaned['industryworkedin'] < 7270)] = 10
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 7270) & (dataToBeCleaned['industryworkedin'] < 7860)] = 11
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 7860) & (dataToBeCleaned['industryworkedin'] < 7970)] = 12
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 7970) & (dataToBeCleaned['industryworkedin'] < 8370)] = 13
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 8370) & (dataToBeCleaned['industryworkedin'] < 8560)] = 14
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 8560) & (dataToBeCleaned['industryworkedin'] < 8770)] = 15
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 8770) & (dataToBeCleaned['industryworkedin'] < 9370)] = 16
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 9370) & (dataToBeCleaned['industryworkedin'] < 9670)] = 17
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] >= 9670) & (dataToBeCleaned['industryworkedin'] <= 9870)] = 18
    dataToBeCleaned['industryworkedin'][(dataToBeCleaned['industryworkedin'] == 9920)] = 0

    # replacing all Nan with 0
    dataToBeCleaned['vehicleoccupancy'].fillna(0, inplace=True)
    dataToBeCleaned['traveltimetowork'].fillna(0, inplace=True)
    dataToBeCleaned['workarrivaltime'].fillna(0, inplace=True)
    dataToBeCleaned['hoursworkperweek'].fillna(0, inplace=True)
    dataToBeCleaned['degreefield'].fillna(0, inplace=True)
    dataToBeCleaned['industryworkedin'].fillna(0, inplace=True)

    dataToBeCleaned['workerclass'] = pd.Categorical(dataToBeCleaned.workerclass)
    dataToBeCleaned['vehicleoccupancy'] = pd.Categorical(dataToBeCleaned.vehicleoccupancy)
    dataToBeCleaned['meansoftransport'] = pd.Categorical(dataToBeCleaned.meansoftransport)
    dataToBeCleaned['marital'] = pd.Categorical(dataToBeCleaned.marital)
    dataToBeCleaned['schoolenrollment'] = pd.Categorical(dataToBeCleaned.schoolenrollment)
    dataToBeCleaned['educationalattain'] = pd.Categorical(dataToBeCleaned.educationalattain)
    dataToBeCleaned['sex'] = pd.Categorical(dataToBeCleaned.sex)
    dataToBeCleaned['workarrivaltime'] = pd.Categorical(dataToBeCleaned.workarrivaltime)
    dataToBeCleaned['degreefield'] = pd.Categorical(dataToBeCleaned.degreefield)
    dataToBeCleaned['industryworkedin'] = pd.Categorical(dataToBeCleaned.industryworkedin)

    # deleting categories which are merged above
    del dataToBeCleaned['vehicleoccupancy']
    del dataToBeCleaned['meansoftransport']
    id_col = dataToBeCleaned['idnum']
    return dataToBeCleaned, id_col


cleaned_df_train, traindata_id  = dataCleaning(df_train)
cleaned_df_test, testdata_id = dataCleaning(df_test)


def OneHotEncoderFunct(dataToBeCleaned):
    category_columns = ['workerclass', 'marital', 'schoolenrollment', 'educationalattain', 'sex', 'workarrivaltime', 'degreefield',
         'industryworkedin']
    encoded_data = pd.get_dummies(dataToBeCleaned, columns=category_columns)

    return encoded_data


encoded_X_train = OneHotEncoderFunct(cleaned_df_train)
encoded_X_test = OneHotEncoderFunct(cleaned_df_test)


def GridSearchCvFunc(encoded_df_X, cleaned_df):
    x_train = encoded_df_X
    # print(X.shape)
    # print(X.isnull().sum())
    y_train = pd.DataFrame(cleaned_df['wages'], columns=['wages'])
    rfc = RandomForestRegressor()
    # rfc = LinearRegression()

    param_grid = {
        'n_estimators': [10, 20,30],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [2,4,6]
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(x_train, y_train)
    print(CV_rfc.best_params_)
    return CV_rfc.best_params_


paras = {}
paras = GridSearchCvFunc(encoded_X_train, df_train)


importances = []

def crossValidation(train_data, train_label, k_fold):
    s = int(len(train_data) / k_fold)
    pred_label_list = []

    for i in range(0, k_fold):
        # print(i)
        test_data = train_data[i * s:][:s]
        td = train_data[:i * s].append(train_data[(i + 1) * s:], ignore_index=True)
        tl = train_label[:i * s].append(train_label[(i + 1) * s:], ignore_index=True)

        print(td.shape)
        print(tl.shape)
        # regressor = LinearRegression()
        # regressor = DecisionTreeRegressor(max_depth=6, max_features='sqrt', min_samples_split=4)
        regressor = RandomForestRegressor(max_depth=6, max_features='sqrt', n_estimators=30)

        # usedRFEFor feature elimination
        selector = RFE(regressor, 70, step=1)
        selector.fit(td, tl)
        testSel = selector
        #to be used when not using rfe
        # regressor.fit(td, tl)
        # y_pred = regressor.predict(test_data)
        y_pred = selector.predict(test_data)
        # print(type(y_pred))

        pred_label_list.append(y_pred)

    return pred_label_list,testSel

del encoded_X_train['idnum']
del encoded_X_train['wages']
X = encoded_X_train
y = pd.DataFrame(cleaned_df_train['wages'], columns = ['wages'])
predictedlist,selector=crossValidation(X,y,8)

def rootmeanerror(predictedlist):
    flat_list = [item for sublist in predictedlist for item in sublist]
    rmse = math.sqrt(mean_squared_error(y, flat_list))
    print(rmse)


rootmeanerror(predictedlist)



missing_cols = set( encoded_X_train.columns ) - set( encoded_X_test.columns )
for c in missing_cols:
    encoded_X_test[c] = 0

encoded_X_test =encoded_X_test[encoded_X_train.columns]
X_test = encoded_X_test

def testPreidctor(X_test,selector):

    testPred = selector.predict(X_test)
    return testPred

testPred = testPreidctor(X_test,selector)


def varimp_df(varimp, traindata):
    for i in range(len(varimp)):
        important_features = pd.Series(data=varimp[i],index=traindata.columns)
        important_features.sort_values(ascending=False,inplace=True)
        print(important_features[0:5])
        print('\n\n')

varimp = importances
traindata = X
varimp_df(varimp, traindata)


#export predictions
output = pd.DataFrame()
output['Id'] = testdata_id
output['Wages'] = testPred
output.to_csv('test_outputs.csv', header=True, index=False)
