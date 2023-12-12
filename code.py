import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mvnorm
df = pd.read_excel("diabetes.xlsx")
x_features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
y = ['Outcome']
X_train, X_test, y_train, y_test = train_test_split(df[x_features], df[y], test_size=0.3, random_state=0)
Outcome_0_indices = (y_train == 0)
Outcome_0_data = X_train[Outcome_0_indices['Outcome']]
Outcome_1_indices = (y_train == 1)
Outcome_1_data = X_train[Outcome_1_indices['Outcome']]
#Prior Probablities
P_prob_0 = len(Outcome_0_data)/len(X_train)
P_prob_1 = len(Outcome_1_data)/len(X_train)

#Find mean of each class
class_0_mean = np.mean(Outcome_0_data,axis=0)
class_1_mean = np.mean(Outcome_1_data,axis=0)

#Find Covariance of each class
class_0_covar = np.cov(Outcome_0_data, rowvar=False)
class_1_covar = np.cov(Outcome_1_data, rowvar=False)

#Find determinant of each class
class_0_det = np.linalg.det(class_0_covar)
class_1_det = np.linalg.det(class_1_covar)

#Find Multivariate guassian density
def MVNORM(x,mean,cov):
    return mvnorm.pdf(x,mean,cov)

predictions = []
for i in range(0,len(X_test)):
    x = X_test.iloc[i,0:].values
    liklihood_class_0 = MVNORM(x,class_0_mean,class_0_covar)
    liklihood_class_1 = MVNORM(x,class_1_mean,class_1_covar)

    post_prob_0 = P_prob_0 * liklihood_class_0
    post_prob_1 = P_prob_1 * liklihood_class_1

    if post_prob_0 >= post_prob_1:
        predictions.append(0)
    else:
        predictions.append(1)

# Calculate accuracy
accuracy = np.mean(predictions == y_test['Outcome'].values)

# Print or use accuracy as needed
print("Accuracy:", accuracy)