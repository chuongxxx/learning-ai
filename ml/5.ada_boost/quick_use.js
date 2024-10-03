import { AdaBoostClassifier } from './model.js';

const x1 = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
];
const y1 = [0, 0, 1, 1];

const x1Test = [
    [1.5, 2.5],
    [4.5, 5.5]
];

const abClassifier = new AdaBoostClassifier({ numStumps: 10 });

abClassifier.fit(x1, y1);
const predictions = abClassifier.predict(x1Test);
console.log('Predictions:', predictions);
