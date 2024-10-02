import { DecisionTreeRegressor } from '../decision_tree/model.js';

export class GradientBoost {
    constructor({ maxDepth = 20, learningRate = 0.1, epochs = 10 }) {
        this.maxDepth = maxDepth;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.trees = [];
        this.yMean = 0;
    }

    train(X, y) {
        this.yMean = y.reduce((a, b) => a + b, 0) / y.length;

        let pred = y.map(() => this.yMean);

        for (let i = 0; i < this.epochs; i++) {
            // const loss = this.#calculateLoss(y, pred);
            // console.log('Epoch:', i, '---', 'Loss:', loss);
            const gradient = this.#calculateGradient(y, pred);

            const tree = new DecisionTreeRegressor({ maxDepth: this.maxDepth });
            tree.fit(X, gradient);

            const predictions = X.map((x) => tree.predict(x));

            pred = pred.map((p, i) => p + this.learningRate * predictions[i]);
            this.trees.push(tree);
        }
    }

    #calculateLoss(y, yHat) {
        let sum = 0;
        for (let i = 0; i < y.length; i++) {
            sum += (y[i] - yHat[i]) ** 2;
        }
        return (1 / y.length) * 0.5 * sum;
    }

    #calculateGradient(y, yHat) {
        const gradient = [];
        for (let i = 0; i < y.length; i++) {
            gradient.push(y[i] - yHat[i]);
        }
        return gradient;
    }

    predict(X) {
        let predictions = [];
        for (let i = 0; i < X.length; i++) {
            let pred = this.yMean;
            for (let j = 0; j < this.trees.length; j++) {
                pred += this.learningRate * this.trees[j].predict(X[i]);
            }
            predictions.push(pred);
        }
        return predictions;
    }
}
