"""
===================================================================
Decision Tree Regression
===================================================================

A 1D regression with decision tree.

The :ref:`decision trees <tree>` is
used to fit a sine curve with addition noisy observation. As a result, it
learns local linear regressions approximating the sine curve.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import EraDecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

np.random.seed(1)

era_samples = 20
num_eras = 2

Xs = []
Ys = []

for i in range(num_eras):
    Xs.append( 2 * np.pi * np.random.rand(era_samples, 1) )
    Ys.append( np.sin(Xs[-1] + np.random.rand()).ravel() + np.random.randn()*3 + np.random.randn(era_samples) )
    
X_train = np.vstack(Xs)
y_train = np.hstack(Ys)

X_test = np.sort( 2 * np.pi * np.random.rand(era_samples*3, 1), axis=0 )
y_test = np.sin(X_test).ravel()

eras = np.concatenate( [ ( np.ones(era_samples) ).astype('int') * n for n in range(num_eras) ])

plt.figure(figsize=(8,8))
plt.scatter(X_train, y_train, s=20, edgecolor="black", c="darkorange", label="data")
plt.scatter(X_test, y_test, s=20, edgecolor="black", c="darkgreen", label="target")
mses = []
corrs = []

configs = [(n, 64) for n in range(1,11,1)]

for max_depth, n_est in configs:
    regr_1 = EraDecisionTreeRegressor(
        max_depth=max_depth,
        criterion="era_squared_error"
    )
    regr_1.fit(X_train, y_train, eras)
    print(regr_1)

    # Predict
    y_1 = regr_1.predict(X_test)
    y_0 = np.sin(X_test)

    # Plot the results
    mse = np.round(np.mean((y_1 - y_0)**2),3)
    mses.append(mse)
    print(mse)
    
    plt.plot(X_test, y_1, label=f"depth={max_depth}, n_est={n_est}, mse={mse}", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
