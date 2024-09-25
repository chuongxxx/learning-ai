import {
    KNNClassifier,
    KNNRegressor,
    KNNRegressorWithKDTree,
    KNNWithKDTreeClassifier,
} from "./model.js";
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

// K-D
const x3 = [
    [1, 2],
    [2, 3],
    [5, 4],
    [9, 6],
    [4, 7],
    [8, 1],
    [7, 2],
];
const y3 = [0, 1, 2, 3, 4, 5, 6];
const x3Test = [
    [1, 1],
    [2, 2],
    [8, 6],
    [8, 3],
];

const knnRegressorWithKDTree = new KNNRegressorWithKDTree({ k: 3 });
knnRegressorWithKDTree.fit(x3, y3);
const regressionPredictionsWithKDTree = knnRegressorWithKDTree.predict(x3Test);
console.log("Predictions:", regressionPredictionsWithKDTree);

const x4 = [
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [6, 7, 8, 9],
    [7, 8, 9, 10],
    [8, 9, 10, 11],
];

const y4 = [0, 0, 0, 1, 1, 1];
const x4Test = [
    [4, 5, 6, 7],
    [7, 8, 9, 10],
];

const knnWithKDTreeClassifier = new KNNWithKDTreeClassifier({ k: 3 });
knnWithKDTreeClassifier.fit(x4, y4);

const knnWithKDTreeClassifierPredictions =
    knnWithKDTreeClassifier.predict(x4Test);

console.log("Predictions:", knnWithKDTreeClassifierPredictions);
