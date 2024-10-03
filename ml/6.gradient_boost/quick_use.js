import { GradientBoost } from './model.js';

const X = [[0], [-1], [2], [3], [4], [5], [6], [7], [8], [9]];
const y = [0, -2, 4, 6, 8, 10, 12, 14, 16, 18];

const gb = new GradientBoost({ epochs: 50, maxDepth: 8 });

gb.train(X, y);

const predictions = gb.predict([[1], [10]]);
console.log('Prediction:', predictions);
