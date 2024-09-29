import { DecisionTreeClassifier } from '../decision_tree/model.js';

export class AdaBoostClassifier {
    constructor({ numStumps }) {
        this.numStumps = numStumps;
        this.models = [];
        this.weights = [];
    }

    fit(X, y) {
        this.weights = this.#initWeights(X);
        for (let i = 0; i < this.numStumps; i++) {
            let stump = new DecisionTreeClassifier({
                maxDepth: 1
            }); // Stump is a simple decision tree
            stump.fit(X, y);

            const predictions = X.map((row) => Number(stump.predict(row)));
            const error = this.#computeError(predictions, y, this.weights);
            // Calculate amount of says
            const alpha = this.#computeAlpha(error);
            // Update weights
            this.weights = this.#updateWeights(
                this.weights,
                alpha,
                predictions,
                y
            );
            this.models.push({ stump, alpha });
        }
    }

    #initWeights(X) {
        return X.map(() => 1 / X.length);
    }

    #updateWeights(weights, alpha, predictions, actual) {
        let newWeights = weights.slice();

        for (let i = 0; i < weights.length; i++) {
            let error = Math.abs(predictions[i] - actual[i]);
            newWeights[i] = weights[i] * Math.exp(alpha * error);
        }

        let sumWeights = newWeights.reduce((a, b) => a + b, 0);
        return newWeights.map((w) => w / sumWeights); // Normalize
    }

    #computeAlpha(error) {
        const epsilon = 1e-10;
        const alpha = 0.5 * Math.log((1 - error + epsilon) / (error + epsilon)); // 0.5 * log((1 - error) / error) Amount of says
        return alpha ? alpha : 0;
    }

    // Compute error function
    #computeError(predictions, actual, weights) {
        if (
            predictions.length !== actual.length ||
            predictions.length !== weights.length
        ) {
            throw new Error('All input arrays must have the same length');
        }

        let error = 0;
        const totalWeight = weights.reduce((a, b) => a + b, 0);

        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] !== actual[i]) {
                error += weights[i];
            }
        }
        return error / totalWeight;
    }

    predict(X) {
        let finalPredictions = Array(X.length).fill(0);

        this.models.forEach(({ stump, alpha }) => {
            let predictions = X.map((row) => Number(stump.predict(row)));
            for (let i = 0; i < predictions.length; i++) {
                finalPredictions[i] += alpha * (predictions[i] === 1 ? 1 : -1);
            }
        });

        return finalPredictions.map((pred) => (pred > 0 ? 1 : 0));
    }
}
