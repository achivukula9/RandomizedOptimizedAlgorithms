import mlrose
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.base import BaseEstimator



startExecutionTime=time.time() 

# Importing the dataset
dataset = pd.read_csv('covtype.csv')
X = dataset.iloc[:,:].values
y = dataset.iloc[:,54].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

'''

# Initialize neural network object and fit object

np.random.seed(3)

nn_model = mlrose.NeuralNetwork(hidden_nodes = [5,10], activation ='relu', 
                                 algorithm ='random_hill_climb', 
                                 max_iters = 1000, bias = True, is_classifier = True, 
                                 learning_rate = 0.1, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100)

nn_model.fit(X_train_scaled, y_train)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print('\n\n\n',y_train_accuracy)

print('\n\n\nTime taken to train ', time.time()-startExecutionTime)

startExecutionTime=time.time() 

# Predict labels for test set and assess accuracy
y_test_pred = nn_model.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print('\n\n\n',y_test_accuracy)
    
weights = nn_model.fitted_weights

print('\n\n\n',weights)

print('\n\n\nTime taken to test ', time.time()-startExecutionTime)

'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)

# summarize history for accuracy
plt.plot(nn_model.history['acc'])
plt.plot(nn_model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



print('\n\n\n\n',nn_model.history['acc'])
# summarize history for loss
plt.plot(nn_model.history['loss'])
plt.plot(nn_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''



