import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
# Load the CSV file into a DataFrame
#no NAN values
df = pd.read_csv(r"C:\Users\aavul\Desktop\project k\Fuel.csv")

#DOING ONE HOT ENCODING
column_names = ['MAKE', 'MODEL', 'VEHICLECLASS','TRANSMISSION','FUELTYPE']
    #Taking out the unique datatypes from each column
valuesmake = df['MAKE'].unique().tolist()
valuesmodel = df['MODEL'].unique().tolist()
valuesvehicleclass = df['VEHICLECLASS'].unique().tolist()
valuestransmission = df['TRANSMISSION'].unique().tolist()
valuesfueltype = df['FUELTYPE'].unique().tolist()
    #making them into a list to easily iterate
tobechanged=[valuesmake,valuesmodel,valuesvehicleclass,valuestransmission,valuesfueltype]
for col in column_names:
    for listofstrings in tobechanged:
        i=0
        for tochange in listofstrings:
             df[col] = df[col].replace(tochange,i)
             i=i+1
            #replacing a repeating string 'a' into 1, repeating string 'b' into 2, repeating string 'c' into 3,.......
#MAKING A GUIDE FOR USERINPUTS FOR NEW DATA
guide={}
#guide
for listofstrings in tobechanged:
        i=0
        for tochange in listofstrings:
            guide.update({tochange: i})
            i=i+1

#AS THE DATA IS VERY ORDERED RANDOM FUNCTION HELPS,RANDOMIZING THE COLUMNS
df_random = df.sample(frac=1, random_state=42)
#####  Y IS THE TARGET VARIABLE MATRIX,,X IS THE FEATURE MATRIX
#SLICING THE DF INTO Y_TRAIN    
y=df_random.iloc[:800,12]#taking the first 800 rows(800th rows excluded) and 12th indexed column is co2 emmissions
actual_y= y.to_frame(name='co2 emmissions train')#making the 1d df into a column matrix
Y=np.array(actual_y)
# print(Y.shape)
            
#SLICING THE DF INTO Y_TEST         
Y_te=df_random.iloc[800:,12]#rows from 800(included)
##### AS I AM USING VSCODE I NEED TO CONVERT THIS 1D ARRAYS TO COLUMN MATRIX
Y_tes= Y_te.to_frame(name='co2 emmissions test')#making the 1d df into a column matrix
Y_test=np.array(Y_tes)
# print(Y_test.shape)

#MAKING THE DATA FRAME READY FOR X_TRAIN=X,X_TEST
#ADDING A COLUMN FULL OF 1'S AT INDEX 0 ********************I AM REMOVING THE YEAR COLUMN AS IT CONSTAND FOR ALL CARS***********************
df_random.insert(0, '1 column', 1)
df_random = df_random.drop(labels='MODELYEAR', axis=1)
#PREPARING X=X_TRAIN,TAKING FIRST 12 COLUMNS TO CREATE A FEATURE MATRIX AND 800 ROWS(800TH ROW EXCLUDED)
x=df_random.iloc[:800,0:12]
X=np.array(x)

#PREPARING X_TEST,TAKING FIRST 12 COLUMNS TO CREATE A FEATURE MATRIX AND 800 ROWS(800TH ROW INCLUDED) TO END
X_te=df_random.iloc[800:,0:12]
X_test=np.array(X_te)

#GIVING INITIAL VALUES FOR COEFF MATRIX
bet = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]   
beta=np.array(bet)
#HYPERPARAMETERS
learning_rate = 0.001
numberofloops = 7

# GRADIENT DECENT PROCESS,TRAINING
for iteration in range(numberofloops):
    #Y_HAT IS THE PREDICTIONS
    y_hat= X.dot(beta) 

    #CALUCULATING THE ERROR,USED IN CALUCLATION OF GRADIENT
    error=(Y)-(y_hat)

    # CALCULATING THE GRADIENT MATRIX TO GO OPPOSITE TO THE GRADIENT AND REACH THE MIN GRADIENT
    gradient = (X.T).dot(error) / len(Y)
    
    #UPDATING THE COEFF MATRIX WHICH MAKE OUR COST FUNC MUCH MINIMAL AND PREDICTIONS MORE ACUURATE
    beta = beta - (learning_rate * gradient)

#TESTING THE MODEL AND PREDICTING THE VALUES
y_predict= X_test.dot(beta)
# print(y_predict.shape)

#USING THE R2 FUNC TO CALUCULATE ACCURACY
r_squared = r2_score(Y_test,y_predict)
print("R-squared:", r_squared)


#USING LIBRARIES
cardata= LinearRegression()
# Fit the model to your data
cardata.fit(X, Y)

predicted_values =cardata.predict(X_test)
# print(predicted_values.shape)
r_squaredlib = r2_score(Y_test, predicted_values)

print("R-squared with library:", r_squaredlib)