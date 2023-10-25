from sklearn.linear_model import LogisticRegression

from helper.data_generator import generate_data
from helper.lib import *
from plotting.plot_simple import plot_heatmap

# instantiate the model (using the default parameters)
X, y = generate_data(100, 2, 10, outlier_both=True)
logreg = LogisticRegression(random_state=16)
# fit the model with data
logreg.fit(X, y)
# Get the coefficients (thetas) for each feature
thetas = logreg.coef_
# Get the intercept
intercept = logreg.intercept_
theta = np.hstack((intercept, thetas[0]))
y_pred = logreg.predict(X)


cnf_matrix = confusion_matrix(y, y_pred)
plot_heatmap(cnf_matrix)