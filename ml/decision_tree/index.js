import { DecisionTreeClasssifier, DecisionTreeRegressor } from './model.js';

const x1 = [
    [1, 1, 7],
    [1, 0, 12],
    [0, 1, 18],
    [0, 1, 35],
    [1, 1, 38],
    [1, 0, 50],
    [0, 0, 83]
];

const y1 = [0, 0, 1, 1, 1, 0, 0];

const modelClassifier = new DecisionTreeClasssifier({
    maxDepth: 5,
    criterion: 'gini'
});

modelClassifier.fit(x1, y1);
const label = modelClassifier.predict([1, 1, 38]);
console.log('Prediction:', label);

const x2 = [
    [10, 25],
    [20, 73],
    [35, 54],
    [5, 12],
    [7, 80]
];

const y2 = [98, 0, 100, 44, 5];

const modelRegressor = new DecisionTreeRegressor({
    maxDepth: 5,
    minLoss: 0.01,
    minLeafSize: 1
});

modelRegressor.fit(x2, y2);
const prediction = modelRegressor.predict([2, 10]);
console.log('Prediction:', prediction);
