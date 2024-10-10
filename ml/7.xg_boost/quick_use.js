import { XGBoostRegressor } from './model.js';

const X_train = [
    [1.5, 2.3, 0.5],
    [2.5, 1.2, 1.5],
    [3.5, 2.8, 2.0],
    [4.5, 3.5, 1.0],
    [1.2, 0.8, 0.1],
    [2.8, 2.2, 0.9],
    [3.0, 2.0, 1.7],
    [4.0, 3.0, 1.2],
    [5.0, 4.5, 2.5],
    [3.2, 1.0, 0.5]
];

const y_train = [0.8, 1.2, 1.5, 2.0, 0.3, 1.0, 1.4, 1.8, 2.5, 1.1];

const X_test = [
    [1.0, 1.0, 0.5],
    [2.0, 2.0, 1.0],
    [3.0, 3.0, 2.0],
    [4.0, 4.0, 2.5],
    [5.0, 5.0, 3.0]
];

const xgbRegressor = new XGBoostRegressor({
    numTrees: 10,
    learningRate: 0.1,
    maxDepth: 5,
    subsample: 0.8,
    lambda: 1.5,
    gamma: 0.0,
    minChildWeight: 25
});

xgbRegressor.fit(X_train, y_train);
xgbRegressor.predict(X_test);
