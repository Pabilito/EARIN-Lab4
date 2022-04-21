import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import timeit
import csv
import datetime

intForRandomState = 42              #Variable to manuipulate the simulation
tSize = 0.25                        #Testing set is 25%, training set is 75% of dataset 

#To display the whole table in console
pd.set_option('display.max_columns', None)

#We create DataFrame by accessing our dataset
#WARNING - relative path
dataset = pd.read_csv("kc_house_data.csv")

#We want to predict values from 'price' column
toPredict = np.array(dataset['price'])

# Remove the data to predict from the dataset
# Axis = 1 means that we will remove a column called 'price'
dataset = dataset.drop('price', axis = 1)

# Preprocessing - convert date to know how old houses are in years
thisYear = datetime.datetime.now().year
dataset['date'] = thisYear - pd.to_datetime(dataset["date"]).dt.year

#Store column labels
metrics_list = list(dataset.columns)

#Store metrics
metrics = np.array(dataset)

#Divide dataset into training and testing set, we can change random_state to change the shuffling of the data
train_metrics, test_metrics, train_toPredict, test_toPredict = train_test_split(metrics, toPredict, test_size = tSize, random_state = intForRandomState)
    
# Instantiate model n decision trees
rf = RandomForestRegressor(random_state = intForRandomState)

#Grid search parameters
parameters_grid = {
   'n_estimators': [100, 200, 500, 1000],                           #Number of trees
   'max_features': ['auto', 'sqrt', 'log2'],                        #Feature selection
   'max_depth' : [3,4,5,6,7,8,9],                                   #Depth of the tree
   'criterion' :['absolute_error', 'squared_error', 'poisson']      #Mean absolute error or mean squre error
}

#We also add cross validation
GSCV = GridSearchCV(estimator=rf, param_grid=parameters_grid, cv=5)  
GSCV.fit(train_metrics, train_toPredict)

start = timeit.default_timer()
# Train the model 
rf.fit(train_metrics, train_toPredict)
stop = timeit.default_timer()
training_time = "{0:0.3f}".format(stop - start)
print('Time to train the model: ', training_time, 's')

# Use the forest's predict method on the test data
predictions = rf.predict(test_metrics)

#Format data in table to 0.3f format | show all entries in arrays
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#Save results to csv file
#WARNING - Relative path on my system
filename = "TestResults/Random forest results with grid search and cross validation.csv"
with open(filename,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=';')
    csvWriter.writerow("Best parameters according to grid search and cross validation:") 
    csvWriter.writerow(GSCV.best_params_) 
    csvWriter.writerow(["Prediction","Actual result"])
    for entry in range (0, len(predictions)):
        csvWriter.writerow([predictions[entry], test_toPredict[entry]])