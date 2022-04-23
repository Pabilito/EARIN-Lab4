import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
'''
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(train_metrics)
y = sc_y.fit_transform(train_toPredict)
'''
X = train_metrics
Y = train_toPredict
regr = SVR(kernel='rbf')
start = timeit.default_timer()
regr.fit(X, Y)
stop = timeit.default_timer()
training_time = "{0:0.3f}".format(stop - start)
print('Time to train the model: ', training_time, 's')
predictions = regr.predict(test_metrics)
'''
regr = make_pipeline(StandardScaler(), LinearSVR(random_state=100, tol=1e-5))

start = timeit.default_timer()
# Train the model 
regr.fit(train_metrics, train_toPredict)
stop = timeit.default_timer()
training_time = "{0:0.3f}".format(stop - start)
print('Time to train the model: ', training_time, 's')
predictions = regr.predict(test_metrics)
'''
#Format data in table to 0.3f format | show all entries in arrays
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#Save results to csv file
#WARNING - Relative path on my system
filename = "TestResults/SVM results.csv"
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
plt.title('SVM Linear Regression')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.legend(['Original value', 'Prediction'])
plt.show()
