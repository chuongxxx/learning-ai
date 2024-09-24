import { KNN } from "./model.js";
const x = [
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
];
const y = [0, 0, 0, 1, 1];

const xTest = [
    [5, 5],
    [8, 8],
];

const knn = new KNN({ k: 3 });

knn.fit(x, y);

const predictions = knn.predict(xTest);
console.log("Predictions:", predictions);

const accuracy = knn.evaluate(xTest, [0, 0]);
console.log("Accuracy:", accuracy);
