
# **INSTALLING TALIB**
"""

url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
!wget $url/libta-lib0_0.4.0-oneiric1_amd64.deb -qO libta.deb
!wget $url/ta-lib0-dev_0.4.0-oneiric1_amd64.deb -qO ta.deb
!dpkg -i libta.deb ta.deb
!pip install ta-lib
import talib

"""# **IMPORTING LIBRARIES**"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline 
import re
import datetime
import math
import talib
#import talib
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from talib import RSI, BBANDS,WILLR,MACD
import re
import talib as ta
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from sklearn.model_selection import GridSearchCV
import random
import pandas as pd
import numpy as np
import talib as ta
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib import rcParams
import random
import seaborn as sns
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from google.colab import files
from sklearn import svm

def Scaler(x):
    a=x-x.min()
    b=x.max()-x.min()
    c=-1+2*a/b
    return np.array(c)

"""# **UPLOADING DATA**"""

uploaded = files.upload()
dataset=pd.read_csv("^NSEI.csv")
dataset.columns = ['Date', 'Open', 'High','Low', 'Price', 'Change','Vol']
dataset.tail()

def format_date(date):
    y = date.split('-')
    x = datetime.datetime(int(y[0]),int(y[1]),int(y[2]))
    return x.strftime("%b %d, %Y")

for data in [dataset]:
    data['Date'] = data['Date'].apply(lambda x: format_date(x))

for data in [dataset]:
    data.set_index('Date', inplace=True)

df = dataset.dropna()
df.tail()
# df.drop('Date',axis = 1,inplace = True)

"""# **ADDING** **INDICATORS** **TO** **OUR** **INITIAL** **DATA**"""

indicators = pd.DataFrame()

indicators['EMA']= talib.MA(df['Price'], timeperiod=10 , matype= talib.MA_Type.EMA)
indicators['WR']= talib.WILLR(df['High'], df['Low'], df['Price'], timeperiod=9 )
indicators['MA10']= talib.MA(df['Price'], timeperiod=10)
indicators['MA50'] = talib.MA(df['Price'], timeperiod=50)
indicators['RSI']= talib.RSI(df['Price'], timeperiod= 10)
indicators['MACD'] = talib.MACD(df['Price'])[0]
indicators['BBAND_upper'] = talib.BBANDS(df['Price'])[0]
indicators['BBAND_lower'] = talib.BBANDS(df['Price'])[2]
indicators['SAR'] = talib.SAR(df['High'], df['Low'])
indicators['CCI'] = talib.CCI(df['High'], df['Low'], df['Price'])
indicators['STOCH'] = talib.STOCH(df['High'], df['Low'], df['Price'])[0]
indicators['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Price'])
indicators['ADX'] = talib.ADX(df['High'], df['Low'], df['Price'])
indicators['ROC'] = talib.ROC(df['Price'])
indicators['ATR'] = talib.ATR(df['High'], df['Low'], df['Price'])

indicators.tail()

dataset_final = df.drop([ 'Open', 'High', 'Low', 'Change', 'Vol'], axis=1)

dataset_final_ind = dataset_final.join(indicators)
dataset_final_ind.dropna(inplace = True)

"""# **CREATING THE MATRIX USING TECHNICAL INDICATORS**"""

sign = []
change = []

UPPER_AND_LOWER = [int(2)]*(len(dataset_final_ind))
MA10_MA50 = [int(2)]*(len(dataset_final_ind))
WR_new = [int(2)]*(len(dataset_final_ind))
SAR_new = [int(2)]*(len(dataset_final_ind))
CCI_new = [int(2)]*(len(dataset_final_ind))
STOCH_new = [int(2)]*(len(dataset_final_ind))
ADX_new = [int(2)]*(len(dataset_final_ind))
ROC_new = [int(2)]*(len(dataset_final_ind))
ATR_new = [int(2)]*(len(dataset_final_ind))
RSI_new = [int(2)]*(len(dataset_final_ind))

for i in range(len(dataset_final_ind.Price)-1):
   
        if dataset_final_ind.WR[i]<-80:
          WR_new[i] = -1
        elif dataset_final_ind.WR[i]>-20:
          WR_new[i] = 1
        else:
          WR_new[i] = int(2)
        
        if dataset_final_ind.MA10[i]> dataset_final_ind.MA50[i]:
          MA10_MA50[i] = int(+1)
        else:
           MA10_MA50[i] = (int(-1))
        
        if dataset_final_ind.SAR[i]< dataset_final_ind.Price[i]:
           SAR_new[i] = int(1)
        else:
           SAR_new[i] = int(-1)
        
        if dataset_final_ind.CCI[i]> 100:
           CCI_new[i] = int(1)
        elif dataset_final_ind.CCI[i] < -100:
           CCI_new[i] = int(-1)
        else:
           CCI_new[i] = int(2)
        
        if dataset_final_ind.STOCH[i]> 80:
           STOCH_new[i] = int(1)
        elif dataset_final_ind.STOCH[i]<20:
           STOCH_new[i] = int(-1)
        else:
            STOCH_new[i] = int(2)
        
        if dataset_final_ind.ADX[i]> 25:
           ADX_new[i] = int(1)
        elif dataset_final_ind.ADX[i]<25:
           ADX_new[i] = int(-1)
        
        if dataset_final_ind.ROC[i]> 0:
           ROC_new[i] = int(1)
        elif dataset_final_ind.ROC[i]<0:
           ROC_new[i] = int(-1)
        
        if dataset_final_ind.ATR[i] + dataset_final_ind.Price[i]>dataset_final_ind.Price[i+1]:
           ATR_new[i] = int(+1)
        else:
           ATR_new[i] = int(-1)

        if dataset_final_ind.RSI[i]>80:
          RSI_new[i] = int(-1)
        elif dataset_final_ind.RSI[i]<20:
          RSI_new[i] = int(1)
        else:
          RSI_new[i] = int(2)

        if dataset_final_ind.Price[i]>dataset_final_ind.BBAND_upper[i]:
          UPPER_AND_LOWER[i] = int(-1)
        if dataset_final_ind.Price[i]<dataset_final_ind.BBAND_lower[i]:
          UPPER_AND_LOWER[i] = int(+1)  

        
        

        
        if dataset_final_ind.Price[i+1] - dataset_final_ind.Price[i] >0:
            value=int(1)
        else:
            value=int(-1)
        sign.append(int(value))

last_day_data = dataset_final_ind.iloc[-1]

"""# **REMOVING LAST DAY DATA**"""

dataset_final_ind.drop(labels = dataset_final_ind.tail(1).index[0], inplace = True)

UPPER_AND_LOWER.pop()
MA10_MA50.pop()
WR_new.pop()
SAR_new.pop()
CCI_new.pop()
STOCH_new.pop()
ADX_new.pop()
ROC_new.pop()
ATR_new.pop()
RSI_new.pop()



dataset_final_ind.fillna(method= 'pad')

dataset_final_ind

returns = []
for i in range(1,len(dataset_final_ind)):
  z = dataset_final_ind.Price[i]-dataset_final_ind.Price[i-1]
  z/=dataset_final_ind.Price[i-1]
  returns.append(z)
print(sum(returns))

"""# **FINAL MATRIX**"""

from collections import defaultdict 
d = {'MA10_MA50':MA10_MA50,'WR_new':WR_new,'SAR_new':SAR_new,'CCI_new':CCI_new,'STOCH_new':STOCH_new,'ADX_new':ADX_new,'ROC_new':ROC_new,'ATR_new':ATR_new,'RSI_new':RSI_new,'sign':sign}
df = pd.DataFrame(d)

"""# **DROPPING INDICATORS HAVING 2**"""

df.drop(['WR_new','CCI_new','STOCH_new','RSI_new'], axis = 1,inplace=True)

df

"""# **LSTM**"""

# splittig training and testing data

df1=df.reset_index(drop=True)
dataset_final_ind = df1.dropna()
data = dataset_final_ind[-2000:]
data
split_ratio = 0.75
data_training = data.iloc[:int(data.shape[0]*split_ratio)]

data_test = data.iloc[int(data.shape[0]*split_ratio):]

data_training.shape, data_test.shape

def standardize(data):    
    from sklearn.preprocessing import MinMaxScaler
    MinMax = MinMaxScaler()
    standardized_data = MinMax.fit_transform(data)
    return standardized_data, MinMax

def split_data(window,data,stock_code):
    X = []
    y = []

    for i in range(window, data.shape[0]):
        X.append(data[i-window:i, 0:-1])
        y.append(int(data[i][-1]))

    X = np.array(X)
    y = np.array(y)

    return X, y

X = []
y = []   
window = 15
training_data, scaler= standardize(data = data_training)
training_data.shape

def def_model(units, dropout, input_shape):
    model_lstm = Sequential()
    model_lstm.add(LSTM(units = units, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model_lstm.add(Dropout(dropout))
    model_lstm.add(LSTM(units = units))
    model_lstm.add(Dense(1))

    return model_lstm

"""# **Compiling** **and** **Predicting** **using** **Indicators**"""

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def RMSE(actual,predicted):
 mse = sklearn.metrics.mean_squared_error(actual, predicted)
 return mse

stock_code = 1
batch_size = 4
dropout = 0.4
epochs= 100
units = 150
window_size = 15


#Standardize Data
training_data, scaler= standardize(data = data_training)

#Split Test Data
X_train, y_train = split_data(window= window_size, data= training_data, stock_code= stock_code)

#Create Model
model_lstm = def_model(units=units, dropout= dropout, input_shape= (8, 1))

#learning_rate = [0.0001, 0.001, 0.005, 0.01]
#Best is 0.005

adam = tf.keras.optimizers.Adam(lr=0.0001)
model_lstm.compile(optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

#Fit the model
history = model_lstm.fit(X_train, y_train, batch_size= batch_size, epochs= epochs, verbose=2)

print('\nAccuracy Graph')
plt.figure(figsize=(25, 5))
from scipy.ndimage.filters import gaussian_filter1d

smooth = gaussian_filter1d(history.history['binary_accuracy'], sigma=2)

plt.plot(smooth, color ='red', label= 'Accuracy', linewidth = 1)
#plt.plot(history.history['val_loss'], color ='blue', label= 'Validation Loss')
plt.title('Accuracy')
plt.xlabel('Time')
plt.legend()
plt.show()

#append the tail of training data in the test data
past_days = data_training.tail(window_size)
df1 = past_days.append(data_test)

#Standardize the test data
test_data = scaler.transform(df1)

#Split the test data
X_test, y_test = split_data(window= window_size, data= test_data, stock_code=stock_code)

#Make Prediction using the Trained model
y_pred = model_lstm.predict(X_test)

ind = []
for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        ind.append(1)
    else:
        ind.append(0)

"""# **ACCURACY**"""

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_matrix(ind, y_test)
print(accuracy_score(ind, y_test))

"""# **ERRORS**"""

mse = mean_squared_error(y_test, ind)
r2 = r2_score(y_test, ind)
print('MAE',':',mse)
print('RMSE',':',mse**(0.5))
print('r^2',':',r2)

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

precision = precision_score(y_test, ind)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, ind)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, ind)
print('F1 score: %f' % f1)

"""## **PCA + GENETIC + LSTM**"""

# Reducing dimensions using PCA to 3 dimensions
df=df.reset_index(drop=True)
featuredata1 = df.dropna()
featuredata1
X=featuredata1[['MA10_MA50'	,'WR_new'	,'SAR_new'	,'CCI_new'	,'STOCH_new'	,'ADX_new'	,'ROC_new'	,'ATR_new'	,'RSI_new']]


from sklearn.decomposition import PCA
pca= PCA(n_components=3)
x_pc=pca.fit_transform(X)

df1=pd.DataFrame(data=x_pc,columns=["PC1","PC2","PC3"])
df1['sign']=featuredata1['sign']
df1

# Genetic code for fitting the model 

RFC = svm.SVC()
RFC=RandomForestClassifier( n_estimators= 100,
 n_jobs=-1)
class GeneticSelector():
    def __init__(self, estimator, n_gen, size, n_best, n_rand, 
                 n_children, mutation_rate):
        # Estimator 
        self.estimator = estimator
        # Number of generations
        self.n_gen = n_gen
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select
        self.n_best = n_best
        # Number of random chromosomes to select
        self.n_rand = n_rand
        # Number of children created during crossover
        self.n_children = n_children
        # Probablity of chromosome mutation
        self.mutation_rate = mutation_rate
        
        #if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:
            #raise ValueError("The population size is not stable.")  
    
    def initialization(self):
        population=[]
        for i in range(self.size):  # size-no. of chromosomes in population
            chromosome=np.ones(self.n_features,dtype=np.bool) #makes an array of true
            mask=np.random.rand(len(chromosome))<0.3 #makes a boolean array when the value of the random int is <0.3 returns true.
            chromosome[mask]=False #when the mask array is having a true then at this position chromosome array is changed from true to false.
            population.append(chromosome)
        return population
    def fitness(self,population):
        scores=[]
        X, y = self.dataset
        X=pd.DataFrame(data=Scaler(X))
        X_train = X[:int(X.shape[0]*0.7)]
        X_test = X[int(X.shape[0]*0.7):]
        y_train = y[:int(X.shape[0]*0.7)]
        y_test = y[int(X.shape[0]*0.7):]
        for chromosome in population:
            try:
                self.estimator.fit(X_train.iloc[:,chromosome],y_train)
                predictions = self.estimator.predict(X_test.iloc[:,chromosome])            
                scores.append(accuracy_score(y_test,predictions))
            except:
                continue
        scores, population = np.array(scores), np.array(population) #to sort the scores list its converted to array
        inds=np.argsort(scores)[::-1] # reversed the order to get indices in descending order of the values.
        return list(scores[inds]), list(population[inds,:]) #sorted scores array is converted into list and the population of chromosomes is arranged according to scores
    
    def select(self, population_sorted):
        population_next = []
        for i in range(self.n_best):
            population_next.append(population_sorted[i]) #these are best selected parents
        for i in range(self.n_rand):
            population_next.append(random.choice(population_sorted)) #these are randomly selected parents
        random.shuffle(population_next)
        return population_next    
    
    def crossover(self,population):  #adding  crossover  method to our class, which mixes the genes of the previously selected   n_best+n_rand  parents.
        population_next=[]
        for i in range(int(len(population)/2)): #is done as 2 parents(chromosomes) are reqd for a child.
            for j in range(self.n_children):
                chromosome1,chromosome2=population[i], population[len(population)-1-i]  #mating criteria is first and last parent from population.-1 is there because index starts from 0 whereas length starts from 1.
                child = chromosome1
                mask = np.random.rand(len(child))>0.5 # with 50% probability genes are mutated
                child[mask]=chromosome2[mask] #when mask is true the chromosome2 boolean values are put into the child with same indexes.
                population_next.append(child)
        return population_next
    
    def mutate(self,population):
        population_next=[]
        for i in range(len(population)):
            chromosome=population[i]
            if random.random()<self.mutation_rate: # 2 probabilites are involved here 1 is self mutation rate and other is 0.05.
                mask=np.random.rand(len(chromosome))<0.05
                chromosome[mask]=False
            population_next.append(chromosome)
        return population_next
    
    def generate(self,population):
        #selection crossover and mutation
        scores_sorted,population_sorted=self.fitness(population)
        population=self.select(population_sorted)
        population=self.crossover(population)
        population=self.mutate(population)
    
        #History ( saves the best results of each generation.)
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))
    
        return population
    
    def fit(self,X,y): #pass data and perform genetic algorithm
        
        self.chromosomes_best = []
        self.scores_best, self.scores_avg  = [], []
        
        self.dataset = X, y
        self.n_features = X.shape[1]
        
        population = self.initialization()
        for i in range(self.n_gen):
            population = self.generate(population)            
        return self
    
    @property
    def support_(self): # returns a chromosome with the best features (the best chromosome from the last generation).
        return self.chromosomes_best[-1] 
 
    def plot_scores(self):
        plt.plot(self.scores_best, label='Best')
        plt.plot(self.scores_avg, label='Average')
        plt.legend(loc="best")
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.show()

df1=df1.reset_index(drop=True)
featuredata1 = df1.dropna()
featuredata1
X=featuredata1[["PC1","PC2","PC3"]]
y=featuredata1["sign"]
featuredata1

sel = GeneticSelector(estimator=RFC,n_gen=20, size=90, n_best=25, n_rand=25, 
                      n_children=25, mutation_rate=0.05)
sel.fit(X,y)

# best features scores
sel.scores_best

# best features
sel.chromosomes_best

best_chromosome = [True,  False, True]

"""# **LSTM**"""

# Using PCA and genetic matrix, splitting the data into training and testing 

df1.drop(['PC2'], axis=1,inplace = True)

df1=df1.reset_index(drop=True)
dataset_final_ind = df1.dropna()


data = dataset_final_ind[-2000:]
data
split_ratio = 0.75
data_training = data.iloc[:int(data.shape[0]*split_ratio)]
# data_training = data_training.iloc[:,best_chromosome]
data_test = data.iloc[int(data.shape[0]*split_ratio):]
# data_test=data_test.iloc[:,best_chromosome]
data_training.shape, data_test.shape

def standardize(data):    
    from sklearn.preprocessing import MinMaxScaler
    MinMax = MinMaxScaler()
    standardized_data = MinMax.fit_transform(data)
    return standardized_data, MinMax

def split_data(window,data,stock_code):
    X = []
    y = []

    for i in range(window, data.shape[0]):
        X.append(data[i-window:i, 0:-1])
        y.append(int(data[i][-1]))

    X = np.array(X)
    y = np.array(y)

    return X, y

X = []
y = []   
window = 20
training_data, scaler= standardize(data = data_training)
training_data.shape

def def_model(units, dropout, input_shape):
    model_lstm = Sequential()
    model_lstm.add(LSTM(units = units, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model_lstm.add(Dropout(dropout))
    model_lstm.add(LSTM(units = units))
    model_lstm.add(Dense(1))

    return model_lstm

"""# **Compiling** **and** **Predicting** **using** **Indicators**"""

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def RMSE(actual,predicted):
 mse = sklearn.metrics.mean_squared_error(actual, predicted)
 return mse

stock_code = 1
batch_size = 4
dropout = 0.4
epochs= 100
units = 150
window_size = 20


#Standardize Data
training_data, scaler= standardize(data = data_training)

#Split Test Data
X_train, y_train = split_data(window= window_size, data= training_data, stock_code= stock_code)

#Create Model
model_lstm = def_model(units=units, dropout= dropout, input_shape= (8, 1))

#learning_rate = [0.0001, 0.001, 0.005, 0.01]
#Best is 0.005

adam = tf.keras.optimizers.Adam(lr=0.0001)
model_lstm.compile(optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

#Fit the model
history = model_lstm.fit(X_train, y_train, batch_size= batch_size, epochs= epochs, verbose=2)

print('\nAccuracy Graph')
plt.figure(figsize=(25, 5))
from scipy.ndimage.filters import gaussian_filter1d

smooth = gaussian_filter1d(history.history['binary_accuracy'], sigma=2)

plt.plot(smooth, color ='red', label= 'Accuracy', linewidth = 1)
#plt.plot(history.history['val_loss'], color ='blue', label= 'Validation Loss')
plt.title('Accuracy')
plt.xlabel('Time')
plt.legend()
plt.show()

#append the tail of training data in the test data
past_days = data_training.tail(window_size)
df2 = past_days.append(data_test)

#Standardize the test data
test_data = scaler.transform(df2)

#Split the test data
X_test, y_test = split_data(window= window_size, data= test_data, stock_code=stock_code)

#Make Prediction using the Trained model
y_pred = model_lstm.predict(X_test)

ind = []
for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        ind.append(1)
    else:
        ind.append(0)

"""# **ACCURACY**"""

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_matrix(ind, y_test)
print(accuracy_score(ind, y_test))

"""# **ERRORS**"""

mse = mean_squared_error(y_test, ind)
r2 = r2_score(y_test, ind)
print('MAE',':',mse)
print('RMSE',':',mse**(0.5))
print('r^2',':',r2)

"""# **PREDICTION OF PRICES**"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

"""# **UPLOADING DATA**"""

uploaded = files.upload()
dataset=pd.read_csv("CHINA.csv")
# dataset.columns = ['Date', 'Open', 'High','Low', 'Price', 'Change','Vol']
dataset.tail()
df = dataset

df_final = df
df_final.shape

df_final.describe()

df_final.dropna(inplace = True)
df_final.isnull().values.any()

df_final['Adj Close'].plot()

test = df_final
# Target column
target_adj_close = pd.DataFrame(test['Adj Close'])

df.drop(['Volume'], axis = 1,inplace=True)
display(test.head())

feature_columns = ['Open', 'High', 'Low']

"""# **NORMALIZING THE DATA**"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
feature_minmax_transform = pd.DataFrame(columns=feature_columns, data=feature_minmax_transform_data, index=test.index)
feature_minmax_transform.head()

display(feature_minmax_transform.head())
print('Shape of features : ', feature_minmax_transform.shape)
print('Shape of target : ', target_adj_close.shape)

# Shift target array because we want to predict the n + 1 day value


target_adj_close = target_adj_close.shift(-1)
validation_y = target_adj_close[-90:-1]
target_adj_close = target_adj_close[:-90]

# Taking last 90 rows of data to be validation set
validation_X = feature_minmax_transform[-90:-1]
feature_minmax_transform = feature_minmax_transform[:-90]
display(validation_X.tail())
display(validation_y.tail())

print("\n -----After process------ \n")
print('Shape of features : ', feature_minmax_transform.shape)
print('Shape of target : ', target_adj_close.shape)
display(target_adj_close.tail())

"""# **TRAIN-TEST-SPLIT**"""

from sklearn.model_selection import train_test_split, cross_val_predict, TimeSeriesSplit, KFold, cross_val_score
ts_split= TimeSeriesSplit(n_splits=10)
for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(), target_adj_close[len(train_index): (len(train_index)+len(test_index))].values.ravel()

X_train.shape

X_test.shape

y_train.shape

y_test.shape

def validate_result(model, model_name):
    predicted = model.predict(validation_X)
    RSME_score = np.sqrt(mean_squared_error(validation_y, predicted))
    print('RMSE: ', RSME_score)
    
    R2_score = r2_score(validation_y, predicted)
    print('R2 score: ', R2_score)

    plt.plot(validation_y.index, predicted,'r', label='Predict')
    plt.plot(validation_y.index, validation_y,'b', label='Actual')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.title(model_name + ' Predict vs Actual')
    plt.legend(loc='upper right')
    plt.show()

X_train =np.array(X_train)
X_test =np.array(X_test)

X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=8, verbose=1, shuffle=False, callbacks=[early_stop])

y_pred_test_lstm = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
r2_train = r2_score(y_train, y_train_pred_lstm)

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
r2_test = r2_score(y_test, y_pred_test_lstm)

mse = mean_squared_error(y_test, y_pred_test_lstm)
r2 = r2_score(y_test, y_pred_test_lstm)
print('MAE',':',mse)
print('RMSE',':',mse**(0.5))
print('r^2',':',r2)

"""# **LSTM ON OPEN HIGH LOW CLOSE**"""

uploaded = files.upload()
dataset=pd.read_csv("nifty.csv")
dataset.columns = ['Date', 'Open', 'High','Low', 'Price', 'Change','Vol']
dataset.tail()

def format_date(date):
    y = date.split('-')
    x = datetime.datetime(int(y[0]),int(y[1]),int(y[2]))
    return x.strftime("%b %d, %Y")

for data in [dataset]:
    data['Date'] = data['Date'].apply(lambda x: format_date(x))

for data in [dataset]:
    data.set_index('Date', inplace=True)

df = dataset.dropna()
df.tail()

indicators = pd.DataFrame()

indicators['EMA']= talib.MA(df['Price'], timeperiod=10 , matype= talib.MA_Type.EMA)
indicators['WR']= talib.WILLR(df['High'], df['Low'], df['Price'], timeperiod=9 )
indicators['MA10']= talib.MA(df['Price'], timeperiod=10)
indicators['MA50'] = talib.MA(df['Price'], timeperiod=50)
indicators['RSI']= talib.RSI(df['Price'], timeperiod= 10)
indicators['MACD'] = talib.MACD(df['Price'])[0]
indicators['BBAND_upper'] = talib.BBANDS(df['Price'])[0]
indicators['BBAND_lower'] = talib.BBANDS(df['Price'])[2]
indicators['SAR'] = talib.SAR(df['High'], df['Low'])
indicators['CCI'] = talib.CCI(df['High'], df['Low'], df['Price'])
indicators['STOCH'] = talib.STOCH(df['High'], df['Low'], df['Price'])[0]
indicators['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Price'])
indicators['ADX'] = talib.ADX(df['High'], df['Low'], df['Price'])
indicators['ROC'] = talib.ROC(df['Price'])
indicators['ATR'] = talib.ATR(df['High'], df['Low'], df['Price'])

indicators.tail()

dataset_final_ind = df.join(indicators)
dataset_final_ind.dropna(inplace = True)
dataset_final_ind

sign = []
change = []

UPPER_AND_LOWER = [int(2)]*(len(dataset_final_ind))
MA10_MA50 = [int(2)]*(len(dataset_final_ind))
WR_new = [int(2)]*(len(dataset_final_ind))
SAR_new = [int(2)]*(len(dataset_final_ind))
CCI_new = [int(2)]*(len(dataset_final_ind))
STOCH_new = [int(2)]*(len(dataset_final_ind))
ADX_new = [int(2)]*(len(dataset_final_ind))
ROC_new = [int(2)]*(len(dataset_final_ind))
ATR_new = [int(2)]*(len(dataset_final_ind))
RSI_new = [int(2)]*(len(dataset_final_ind))
HIGH_new = [int(1)]*(len(dataset_final_ind))
OPEN_new = [int(1)]*(len(dataset_final_ind))
LOW_new = [int(1)]*(len(dataset_final_ind))

for i in range(len(dataset_final_ind.Price)-1):
        if i>0:
          if dataset_final_ind.High[i]> dataset_final_ind.High[i-1]:
            HIGH_new[i] = 1
          else:
            HIGH_new[i] = -1
          if dataset_final_ind.Open[i]> dataset_final_ind.Open[i-1]:
            OPEN_new[i] = 1
          else:
            OPEN_new[i] = -1
          if dataset_final_ind.Low[i]> dataset_final_ind.Low[i-1]:
            LOW_new[i] = 1
          else:
            LOW_new[i] = -1
          

       
        if dataset_final_ind.WR[i]<-80:
          WR_new[i] = -1
        elif dataset_final_ind.WR[i]>-20:
          WR_new[i] = 1
        else:
          WR_new[i] = int(2)
        
        if dataset_final_ind.MA10[i]> dataset_final_ind.MA50[i]:
          MA10_MA50[i] = int(+1)
        else:
           MA10_MA50[i] = (int(-1))
        
        if dataset_final_ind.SAR[i]< dataset_final_ind.Price[i]:
           SAR_new[i] = int(1)
        else:
           SAR_new[i] = int(-1)
        
        if dataset_final_ind.CCI[i]> 100:
           CCI_new[i] = int(1)
        elif dataset_final_ind.CCI[i] < -100:
           CCI_new[i] = int(-1)
        else:
           CCI_new[i] = int(2)
        
        if dataset_final_ind.STOCH[i]> 80:
           STOCH_new[i] = int(1)
        elif dataset_final_ind.STOCH[i]<20:
           STOCH_new[i] = int(-1)
        else:
            STOCH_new[i] = int(2)
        
        if dataset_final_ind.ADX[i]> 25:
           ADX_new[i] = int(1)
        elif dataset_final_ind.ADX[i]<25:
           ADX_new[i] = int(-1)
        
        if dataset_final_ind.ROC[i]> 0:
           ROC_new[i] = int(1)
        elif dataset_final_ind.ROC[i]<0:
           ROC_new[i] = int(-1)
        
        if dataset_final_ind.ATR[i] + dataset_final_ind.Price[i]>dataset_final_ind.Price[i+1]:
           ATR_new[i] = int(+1)
        else:
           ATR_new[i] = int(-1)

        if dataset_final_ind.RSI[i]>80:
          RSI_new[i] = int(-1)
        elif dataset_final_ind.RSI[i]<20:
          RSI_new[i] = int(1)
        else:
          RSI_new[i] = int(2)

        if dataset_final_ind.Price[i]>dataset_final_ind.BBAND_upper[i]:
          UPPER_AND_LOWER[i] = int(-1)
        if dataset_final_ind.Price[i]<dataset_final_ind.BBAND_lower[i]:
          UPPER_AND_LOWER[i] = int(+1)  

        
        

        
        if dataset_final_ind.Price[i+1] - dataset_final_ind.Price[i] >0:
            value=int(1)
        else:
            value=int(-1)
        sign.append(int(value))

last_day_data = dataset_final_ind.iloc[-1]

"""# **REMOVING LAST DAY DATA**"""

dataset_final_ind.drop(labels = dataset_final_ind.tail(1).index[0], inplace = True)

UPPER_AND_LOWER.pop()
MA10_MA50.pop()
WR_new.pop()
SAR_new.pop()
CCI_new.pop()
STOCH_new.pop()
ADX_new.pop()
ROC_new.pop()
ATR_new.pop()
RSI_new.pop()
HIGH_new.pop()
OPEN_new.pop()
LOW_new.pop()




dataset_final_ind.fillna(method= 'pad')

dataset_final_ind

returns = []
for i in range(1,len(dataset_final_ind)):
  z = dataset_final_ind.Price[i]-dataset_final_ind.Price[i-1]
  z/=dataset_final_ind.Price[i-1]
  returns.append(z)
print(sum(returns))

"""# **FINAL MATRIX**"""

# Generating three matrixes one for OPHL and one combining with technical indicators
from collections import defaultdict 
d1 = {'MA10_MA50':MA10_MA50,'WR_new':WR_new,'SAR_new':SAR_new,'CCI_new':CCI_new,'STOCH_new':STOCH_new,'ADX_new':ADX_new,'ROC_new':ROC_new,'ATR_new':ATR_new,'RSI_new':RSI_new,'sign':sign}

df1 = pd.DataFrame(d1)

d2 = {'OPEN_new':OPEN_new,'HIGH_new':HIGH_new,'LOW_new':LOW_new,'sign':sign}

df2 = pd.DataFrame(d2)

d3 = {'MA10_MA50':MA10_MA50,'WR_new':WR_new,'SAR_new':SAR_new,'CCI_new':CCI_new,'STOCH_new':STOCH_new,'ADX_new':ADX_new,'ROC_new':ROC_new,'ATR_new':ATR_new,'RSI_new':RSI_new,'sign':sign,'OPEN_new':OPEN_new,'HIGH_new':HIGH_new,'LOW_new':LOW_new}

df3 = pd.DataFrame(d3)

# splittig training and testing data

df3=df3.reset_index(drop=True)
dataset_final_ind = df3.dropna()
data = dataset_final_ind[-2000:]
data
split_ratio = 0.75
data_training = data.iloc[:int(data.shape[0]*split_ratio)]

data_test = data.iloc[int(data.shape[0]*split_ratio):]

data_training.shape, data_test.shape

def standardize(data):    
    from sklearn.preprocessing import MinMaxScaler
    MinMax = MinMaxScaler()
    standardized_data = MinMax.fit_transform(data)
    return standardized_data, MinMax

def split_data(window,data,stock_code):
    X = []
    y = []

    for i in range(window, data.shape[0]):
        X.append(data[i-window:i, 0:-1])
        y.append(int(data[i][-1]))

    X = np.array(X)
    y = np.array(y)

    return X, y

X = []
y = []   
window = 15
training_data, scaler= standardize(data = data_training)
training_data.shape

def def_model(units, dropout, input_shape):
    model_lstm = Sequential()
    model_lstm.add(LSTM(units = units, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model_lstm.add(Dropout(dropout))
    model_lstm.add(LSTM(units = units))
    model_lstm.add(Dense(1))

    return model_lstm

"""# **Compiling** **and** **Predicting** **using** **Indicators**"""

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def RMSE(actual,predicted):
 mse = sklearn.metrics.mean_squared_error(actual, predicted)
 return mse

stock_code = 1
batch_size = 4
dropout = 0.4
epochs= 100
units = 150
window_size = 15


#Standardize Data
training_data, scaler= standardize(data = data_training)

#Split Test Data
X_train, y_train = split_data(window= window_size, data= training_data, stock_code= stock_code)

#Create Model
model_lstm = def_model(units=units, dropout= dropout, input_shape= (8, 1))

#learning_rate = [0.0001, 0.001, 0.005, 0.01]
#Best is 0.005

adam = tf.keras.optimizers.Adam(lr=0.0001)
model_lstm.compile(optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

#Fit the model
history = model_lstm.fit(X_train, y_train, batch_size= batch_size, epochs= epochs, verbose=2)

print('\nAccuracy Graph')
plt.figure(figsize=(25, 5))
from scipy.ndimage.filters import gaussian_filter1d

smooth = gaussian_filter1d(history.history['binary_accuracy'], sigma=2)

plt.plot(smooth, color ='red', label= 'Accuracy', linewidth = 1)
#plt.plot(history.history['val_loss'], color ='blue', label= 'Validation Loss')
plt.title('Accuracy')
plt.xlabel('Time')
plt.legend()
plt.show()

#append the tail of training data in the test data
past_days = data_training.tail(window_size)
df1 = past_days.append(data_test)

#Standardize the test data
test_data = scaler.transform(df1)

#Split the test data
X_test, y_test = split_data(window= window_size, data= test_data, stock_code=stock_code)

#Make Prediction using the Trained model
y_pred = model_lstm.predict(X_test)

ind = []
for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        ind.append(1)
    else:
        ind.append(0)

"""# **ACCURACY**"""

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_matrix(ind, y_test)
print(accuracy_score(ind, y_test)*100)

"""# **Fast Fourier Transformation**

"""

######. Importing Libraries
######
!pip install pandas_datareader
import numpy as np
import pylab as pl
from numpy import fft
from datetime import datetime
from pandas_datareader import data as pdr
from google.colab import files

####### Uploading CSV file over here
#######

uploaded = files.upload()
df_original=pd.read_csv("nifty.csv")
data = df_original

####### creating train test and testing set
#######
hist = data.loc[:,'Adj Close'].values
train = data.loc[:'2020-11-01','Adj Close'].values

def fourierExtrapolation(x, n_predict): ###Fourier function
    n = x.size
    n_harm = 51  # Try other values # may be small values
    for i in range(len(x)):
      if np.isnan(x[i]):
        x[i]=x[i-1]
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)
    x_notrend = x - p[0] * t
    x_freqdom = fft.fft(x_notrend)
    f = fft.fftfreq(n)
    indexes = list(range(n))
    indexes.sort(key=lambda i: np.absolute(f[i]))

    #indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
    #indexes.reverse()
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n
        phase = np.angle(x_freqdom[i])
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t
    

###Plotting the data

n_predict = len(hist) - len(train)
extrapolation = fourierExtrapolation(train, n_predict)
pl.plot(np.arange(0, hist.size), hist, 'b', label = 'Data', linewidth = 3)
pl.plot(np.arange(0, train.size), train, 'c', label = 'Train', linewidth = 2)
pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'Predict', linewidth =1)

pl.legend()
pl.show()

"""# **SVM**


"""

import pandas as pd
# !pip install quandl
from google.colab import files
import quandl,datetime
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split #scale, regresions, cross shuffle stats sepeareate data
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style

"""# **FUNCTION FOR GETTING DATA AND PREDICTION**"""

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = [] 
prices = [] 

# Function for getting Data
#########

def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

# Function for Predcition of Prices
#########

def predict_prices(dates,prices,x):
	dates = np.reshape(dates,(len(dates),1))
	svr_len = SVR(kernel='linear' , C= 1e3)
	svr_ploy = SVR(kernel = 'poly', C =1e2 , degree = 2)
	svr_rbf = SVR(kernel = 'rbf' , C = 1e3 , Gamma = 0.1)
	svr_len.fit(dates,prices)
	svr_ploy.fit(dates,prices)
	svr_rbf.fit(dates,prices)

	plt.scatter(dates,prices,color='black',label = 'Data')
	plt.plot(dates,svr_rbf.predict(dates),color ='red', label = 'RBF MODEL')
	plt.plot(dates,svr_len.predict(dates),color = 'green' , label = 'LINEAR MODEL')
	plt.plot(dates,svr_ploy.predict(dates),color = 'blue' , label = 'POLYNOMIAL MODEL')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(' Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0],svr_len.predict(x)[0],svr_ploy.predict(x)[0]

style.use('ggplot')

uploaded = files.upload()
dataset=pd.read_csv("nifty.csv")

df = dataset
df = df.dropna()
df

# Normalize the Data
#########

df['High_Low_per'] = (df['High'] - df['Close']) / df['Close']*100

# Normalize the Data
#########

df['Per_change'] = (df['Open'] - df['Close']) / df['Close']*100

# Set up the parameters
#########

df = df[['Adj Close','High_Low_per','Per_change','Volume']]

label_col = 'Adj Close'
df

forecast_ceil = int(math.ceil(0.001*len(df)))

df['label'] = df[label_col].shift(-forecast_ceil)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_ceil:]
X_lately = X[-forecast_ceil:]

df.dropna(inplace=True)

y = np.array(df['label'])

len(X)

len(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = svm.SVR(kernel='rbf') #svm.SVR()

clf.fit(X_train, y_train)
