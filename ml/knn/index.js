import { KNNClassifier, KNNRegressor } from "./model.js";
const x1 = [
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
];
const y1 = [0, 0, 0, 1, 1];

const x1Test = [
    [5, 5],
    [8, 8],
];

const knnClassifier = new KNNClassifier({ k: 3 });

knnClassifier.fit(x1, y1);

const classPredictions = knnClassifier.predict(x1Test);
console.log("Predictions:", classPredictions);

const x2 = [
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
];
const y2 = [3, 6, 7, 9];

const x2Test = [[3, 3]];
const knnRegressor = new KNNRegressor({ k: 2 });

knnRegressor.fit(x2, y2);

const regressionPredictions = knnRegressor.predict(x2Test);
console.log("Predictions:", regressionPredictions);
