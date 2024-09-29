import {
    DecisionTreeClassifier,
    DecisionTreeRegressor
} from '../decision_tree/model.js';

// Random Forest Model
export class RandomForestClassifier {
    constructor({ numTrees = 10, maxDepth = 5, criterion = 'gini' }) {
        this.numTrees = numTrees;
        this.trees = [];
        this.maxDepth = maxDepth;
        this.criterion = criterion;
    }

    // Fit the Random Forest with multiple decision trees
    fit(x, y) {
        for (let i = 0; i < this.numTrees; i++) {
            const { sampledX, sampledY } = this.#bootstrapSample(x, y);
            const tree = new DecisionTreeClassifier({
                maxDepth: this.maxDepth,
                criterion: this.criterion
            });
            tree.fit(sampledX, sampledY);
            this.trees.push(tree);
        }
    }

    // Predict using majority voting from all trees
    predict(features) {
        const predictions = this.trees.map((tree) => tree.predict(features));
        return this.#majorityVote(predictions);
    }

    // Bootstrap sampling (sample with replacement)
    #bootstrapSample(x, y) {
        const sampledX = [];
        const sampledY = [];
        const n = x.length;
        for (let i = 0; i < n; i++) {
            const index = Math.floor(Math.random() * n);
            sampledX.push(x[index]);
            sampledY.push(y[index]);
        }
        return { sampledX, sampledY };
    }

    // Majority vote across the predictions of multiple trees
    #majorityVote(predictions) {
        const counts = {};
        predictions.forEach((prediction) => {
            counts[prediction] = (counts[prediction] || 0) + 1;
        });
        return Object.keys(counts).reduce((a, b) =>
            counts[a] > counts[b] ? a : b
        );
    }
}

export class RandomForestRegressor {
    constructor({
        numTrees = 10,
        maxDepth = 5,
        minLoss = 0.01,
        minLeafSize = 1
    }) {
        this.numTrees = numTrees;
        this.trees = [];
        this.maxDepth = maxDepth;
        this.minLoss = minLoss;
        this.minLeafSize = minLeafSize;
    }

    fit(x, y) {
        for (let i = 0; i < this.numTrees; i++) {
            const { sampledX, sampledY } = this.#bootstrapSample(x, y);
            const tree = new DecisionTreeRegressor({
                depth: 0,
                maxDepth: this.maxDepth,
                minLoss: this.minLoss,
                minLeafSize: this.minLeafSize
            });
            tree.fit(sampledX, sampledY);
            this.trees = [...this.trees, tree];
        }
    }

    predict(features) {
        const predictions = this.trees.map((tree) => tree.predict(features));
        return (
            predictions.reduce((sum, value) => sum + value, 0) /
            predictions.length
        ); // Average prediction
    }

    #bootstrapSample(x, y) {
        const sampledX = [];
        const sampledY = [];
        const n = x.length;
        for (let i = 0; i < n; i++) {
            const index = Math.floor(Math.random() * n);
            sampledX.push(x[index]);
            sampledY.push(y[index]);
        }
        return { sampledX, sampledY };
    }
}
