import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
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
start = timeit.default_timer()
parameters_grid = {
    'n_estimators': [100, 200],                                      #Number of trees
    'max_features': ['auto', 'sqrt'],                                #Feature selection
    'criterion' :['absolute_error', 'squared_error']                 #Mean absolute error or mean squre error
} #Should be much more, but it will take too much time

#We also add cross validation 5 times
GSCV = GridSearchCV(estimator=rf, param_grid=parameters_grid, cv=5)

# Train the model   
GSCV.fit(train_metrics, train_toPredict)
stop = timeit.default_timer()
training_time = "{0:0.3f}".format(stop - start)
print('Time to train the model: ', training_time, 's')

# Use the forest's predict method on the test data
predictions = GSCV.predict(test_metrics)

#Format data in table to 0.3f format | show all entries in arrays
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#Save results to csv file
#WARNING - Relative path on my system
filename = "TestResults/Random forest results with grid search and cross validation.csv"
with open(filename,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=';')
    csvWriter.writerow(['Mean_absolute_error: ', mean_absolute_error(test_toPredict, predictions)])
    csvWriter.writerow(['Mean_squared_error: ', mean_squared_error(test_toPredict, predictions)])
    csvWriter.writerow(["Prediction","Actual result"])
    for entry in range (0, len(predictions)):
        csvWriter.writerow([predictions[entry], test_toPredict[entry]])

#Graph1 - Price vs Sqft_living 
size = test_metrics[:,4]
plt.scatter(size, test_toPredict, color = 'red', s=0.1)
plt.scatter(size, predictions, color = 'green', s=0.1)
plt.title('Random Forest Regression')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.legend(['Original value', 'Prediction'])
plt.show()

#Graph2 - Price vs Sqft_living, less data 
plt.scatter(size[0:50], test_toPredict[0:50], color = 'red', s=2)
plt.scatter(size[0:50], predictions[0:50], color = 'green', s=2)
plt.title('Random Forest Regression')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.legend(['Original value', 'Prediction'])
plt.show()