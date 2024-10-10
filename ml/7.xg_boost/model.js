import { MSE } from '../common/loss.js';

class DecisionTreeRegressor {
    constructor({
        maxDepth = 5,
        subsample = 1.0,
        minChildWeight = 1.0,
        lambda = 1.0,
        gamma = 0.0,
        idxs = null
    }) {
        this.maxDepth = maxDepth;
        this.subsample = subsample;
        this.minChildWeight = minChildWeight;
        this.lambda = lambda;
        this.gamma = gamma;

        this.idxs = idxs;

        this.value = 0;
        this.bestScore = 0;

        this.left = null;
        this.right = null;
    }

    train(X, gradients, hessians) {
        this.X = X;
        this.gradients = gradients;
        this.hessians = hessians;

        if (!this.idxs) {
            this.idxs = [...Array(this.X.length).keys()];
        }

        // Tính toán giá trị ban đầu
        this.value =
            -gradients.reduce((acc, gi) => acc + gi, 0) /
            (hessians.reduce((acc, hi) => acc + hi, 0) + this.lambda);

        if (this.maxDepth > 0) {
            for (let i = 0; i < this.X[0].length; i++) {
                this.#findBetterSplit(i);
            }

            if (this.#isLeaf()) return;

            const [leftIdxs, rightIdxs] = this.#splitIndexes();
            this.left = new DecisionTreeRegressor({
                maxDepth: this.maxDepth - 1,
                subsample: this.subsample,
                minChildWeight: this.minChildWeight,
                lambda: this.lambda,
                gamma: this.gamma,
                idxs: leftIdxs
            });
            this.right = new DecisionTreeRegressor({
                maxDepth: this.maxDepth - 1,
                subsample: this.subsample,
                minChildWeight: this.minChildWeight,
                lambda: this.lambda,
                gamma: this.gamma,
                idxs: rightIdxs
            });

            // Gọi train cho child nodes
            this.left.train(this.X, this.gradients, this.hessians);
            this.right.train(this.X, this.gradients, this.hessians);
        }
    }

    #findBetterSplit(featureIdx) {
        const featureValues = this.idxs.map((idx) => this.X[idx][featureIdx]);
        const sortedIdx = [...featureValues.keys()].sort(
            (a, b) => featureValues[a] - featureValues[b]
        );

        let gradientLeft = 0;
        let hessianLeft = 0;
        let gradientRight = this.gradients.reduce((acc, gi) => acc + gi, 0);
        let hessianRight = this.hessians.reduce((acc, hi) => acc + hi, 0);

        for (let i = 0; i < sortedIdx.length - 1; i++) {
            const idx = sortedIdx[i];

            const gi = this.gradients[this.idxs[idx]];
            const hi = this.hessians[this.idxs[idx]];

            gradientLeft += gi;
            hessianLeft += hi;
            gradientRight -= gi;
            hessianRight -= hi;

            const [currentFeatureValue, nextFeatureValue] = [
                featureValues[sortedIdx[i]],
                featureValues[sortedIdx[i + 1]]
            ];

            if (
                hessianLeft < this.minChildWeight ||
                hessianRight < this.minChildWeight ||
                currentFeatureValue === nextFeatureValue
            )
                continue;

            const gain = this.#calcGain(
                gradientLeft,
                hessianLeft,
                gradientRight,
                hessianRight
            );
            if (gain > this.bestScore) {
                this.bestScore = gain;
                this.splitFeatureIdx = featureIdx;
                this.threshold = (currentFeatureValue + nextFeatureValue) / 2;
            }
        }
    }

    #calcGain(gradientLeft, hessianLeft, gradientRight, hessianRight) {
        return (
            0.5 *
                (gradientLeft ** 2 / (hessianLeft + this.lambda) +
                    gradientRight ** 2 / (hessianRight + this.lambda)) -
            (gradientLeft + gradientRight) ** 2 /
                (hessianLeft + hessianRight + this.lambda) -
            this.gamma / 2
        );
    }

    #splitIndexes() {
        const splitFeatureValues = this.idxs.map(
            (idx) => this.X[idx][this.splitFeatureIdx]
        );
        return [[], []].map((arr) =>
            splitFeatureValues.forEach((val, idx) =>
                val <= this.threshold ? arr.push(this.idxs[idx]) : null
            )
        );
    }

    #isLeaf() {
        return this.bestScore === 0;
    }

    predict(row) {
        return this.#isLeaf()
            ? this.value
            : row[this.splitFeatureIdx] <= this.threshold
              ? this.left.predict(row)
              : this.right.predict(row);
    }

    predictRows(X) {
        return X.map((row) => this.predict(row));
    }
}
export class XGBoostRegressor {
    constructor({
        numTrees = 10,
        learningRate = 0.3,
        maxDepth = 5,
        subsample = 1.0,
        lambda = 0.0,
        gamma = 0.0,
        minChildWeight = 1.0
    }) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.learningRate = learningRate;
        this.subsample = subsample;
        this.lambda = lambda;
        this.gamma = gamma;
        this.minChildWeight = minChildWeight;

        this.yMean = null;
        this.models = [];
    }

    fit(X, y, verbose = false) {
        this.yMean = y.reduce((a, b) => a + b, 0) / y.length;

        let currentPredictions = Array(y.length).fill(this.yMean);

        for (let i = 0; i < this.numTrees; i++) {
            const gradients = MSE.gradient(currentPredictions, y);
            const hessians = MSE.hessian(currentPredictions, y);

            const sampleIdxs =
                this.subsample < 1.0
                    ? [...Array(X.length).keys()]
                          .sort(() => 0.5 - Math.random())
                          .slice(0, Math.floor(this.subsample * X.length))
                    : null;

            const tree = new DecisionTreeRegressor({
                maxDepth: this.maxDepth,
                subsample: this.subsample,
                minChildWeight: this.minChildWeight,
                lambda: this.lambda,
                gamma: this.gamma,
                idxs: sampleIdxs
            });
            tree.train(X, gradients, hessians);

            currentPredictions = currentPredictions.map(
                (pred, j) => pred + this.learningRate * tree.predictRows(X)[j]
            );

            this.models.push(tree);
            if (verbose)
                console.log(`[${i}] Train loss = ${MSE.loss(this.yMean, y)}`);
        }
    }

    predict(X) {
        return this.models.reduce(
            (preds, model) =>
                preds.map(
                    (pred, i) =>
                        pred + this.learningRate * model.predictRows(X)[i]
                ),
            Array(X.length).fill(this.yMean)
        );
    }
}
