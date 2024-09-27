import { DecisionTree } from './model.js';

const x = [
    [1, 1, 7],
    [1, 0, 12],
    [0, 1, 18],
    [0, 1, 35],
    [1, 1, 38],
    [1, 0, 50],
    [0, 0, 83]
];

const y = [0, 0, 1, 1, 1, 0, 0];

const model = new DecisionTree(3, 'gini');

model.fit(x, y);
const label = model.predict([1, 1, 38]);
console.log('Prediction:', label);
