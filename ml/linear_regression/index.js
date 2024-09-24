import {
    LinearRegressionOLS,
    LinearRegressionGD1,
    LinearRegressionGD2,
    LinearRegressionSGD,
    LinearRegressionNN,
} from "./model.js";

const x = [[1], [2], [4]];
const y = [2, 4, 8];
const model1 = new LinearRegressionOLS();
model1.fit(x, y);
model1.predict([1.5]).print();

const model2 = new LinearRegressionGD1({ learningRate: 0.001, epochs: 70 });
model2.fit(x, y);
model2.predict([1.5]).print();

const model3 = new LinearRegressionGD2({ learningRate: 0.001, epochs: 100 });
model3.fit(x, y);
model3.predict([1.5]).print();

const model4 = new LinearRegressionSGD({ learningRate: 0.001, epochs: 140 });
model4.fit(x, y);
model4.predict([1.5]).print();

const model5 = new LinearRegressionNN({ learningRate: 0.001, epochs: 140 });
await model5.fit(x, y);
model5.predict([1.5]).print();
