#------------------------------------------------------------------------------
# Titanic predictions
#------------------------------------------------------------------------------
# General Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn libraries
from sklearn.preprocessing   import MinMaxScaler, StandardScaler
from sklearn.preprocessing   import Imputer
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import confusion_matrix

# Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.models as km

#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
# Loading data
traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')
X_train = traindata.iloc[:, [2,4,5,6,9]].values  # macht 9 Sinn?
y_train = traindata.iloc[:, 1].values

# Missing data in 'age'
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # hier noch andere methode prüfen (nan weglassen?)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])

# Encoding caterogical data
le = LabelEncoder()
X_train[:,1] = le.fit_transform(X_train[:, 1])
ohe = OneHotEncoder(categorical_features = [1])
X_train = ohe.fit_transform(X_train).toarray()
X_train = X_train[: , 1:]

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#------------------------------------------------------------------------------
# Building the ANN
#------------------------------------------------------------------------------
# Function for ANNs
def make_my_classifier(optimizer, neurons, n_layer):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
    if n_layer > 1: 
        for i in range(n_layer - 1): 
            classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Training of the ANN
classifier = make_my_classifier('rmsprop', 5, 2)
classifier.fit(X_train, y_train, batch_size = 16, epochs = 600)

#------------------------------------------------------------------------------
# Building Random Forest model
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)

#------------------------------------------------------------------------------
# Building XGBoost model
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Result tuning
#------------------------------------------------------------------------------
# Grid search
classifier = KerasClassifier(build_fn = make_my_classifier)
parameters = {'batch_size': [16],
              'epochs': [256],
              'optimizer': ['rmsprop'],
              'neurons': [5],
              'n_layer': [2]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#------------------------------------------------------------------------------
# Predictions
#------------------------------------------------------------------------------
# test data
X_test = testdata.iloc[:, [1,3,4,5,8]].values  

# Missing data in 'age'
imputer_2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # hier noch andere methode prüfen (nan weglassen?)
imputer_2 = imputer_2.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])

imputer_3 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # hier noch andere methode prüfen (nan weglassen?)
imputer_3 = imputer_3.fit(X_test[:, 4:5])
X_test[:, 4:5] = imputer_3.transform(X_test[:, 4:5])

# Encoding caterogical data
le_2 = LabelEncoder()
X_test[:,1] = le_2.fit_transform(X_test[:, 1])
X_test = X_test[:, 0:5]
ohe_2 = OneHotEncoder(categorical_features = [1])
X_test = ohe_2.fit_transform(X_test).toarray()
X_test = X_test[: , 1:]

# Feature scaling
X_test = sc.fit_transform(X_test)

prediction = classifier.predict(X_test)
prediction = (prediction > 0.5)

for i in range(len(prediction)): 
    if prediction[i,0] >= 0.5:
        prediction[i,0] = 1.
    else:
        prediction[i,0] = 0.

cm = confusion_matrix(y_pred_rf, prediction)

y_pred_ann = y_pred_rf
for i in range(len(y_pred_ann)):
    y_pred_ann[i] = int(prediction[i,0])
    
#------------------------------------------------------------------------------
# Save and make ready for submission
#------------------------------------------------------------------------------
output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': y_pred_ann})
output_2 = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': y_pred_rf})

output.to_csv('my_submission.csv', index=False)

submission = pd.read_csv('gender_submission.csv')


