import { RandomForestClassifier, RandomForestRegressor } from './model.js';


const x1 = [
    [1, 2], [2, 3], [3, 1], [6, 8], [7, 9], [8, 6]
];
const y1 = [0, 0, 0, 1, 1, 1];

const rfClassifier = new RandomForestClassifier({ numTrees: 10, maxDepth: 5, criterion: 'gini' });
rfClassifier.fit(x1, y1);
const predictions = rfClassifier.predict([[5, 5], [8, 8]])

console.log('Predictions:', predictions)


const x2 = [
    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
];
const y2 = [1.1, 2.0, 3.1, 3.9, 5.1, 5.9, 6.8, 8.1, 8.9, 10.0];

const rfRegressor = new RandomForestRegressor({ numTrees: 10, maxDepth: 5, minLoss: 0.01, minLeafSize: 1 });
rfRegressor.fit(x2, y2);
const predictions2 = rfRegressor.predict([[5], [8]])

console.log('Predictions:', predictions2)