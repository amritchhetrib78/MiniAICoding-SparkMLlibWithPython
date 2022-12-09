import numpy as np
from sklearn.linear_model import LinearRegression
X_iValues = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y_dValues = 1 * x_0 + 2 * x_1 + pow(X_iValues,2)
y_dValues = np.dot(X_iValues, np.array([1, 2])) + 2 ** X_iValues[3][0]
# X_iValues1 = 1 * 1 + 2 * 1 + 4=7
# y_dValues2 = 1 * 1 + 2 * 2 + 4=9
# y_dValues3 = 2+4+4=10
#y_dValues4  = 2+6+4=12
print("X", X_iValues)
print("Y", y_dValues)
# Training Model with fit() function!
regModel = LinearRegression().fit(X_iValues, y_dValues)
print("Training Result:", regModel)
# Checking accuracy
print("Scores", regModel.score(X_iValues, y_dValues))
print("Predicting Result:", regModel.predict(np.array([[100, 100]])))
