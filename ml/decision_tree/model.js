import { calculateSplitImpurity } from './criteria.js';
import { gini, entropy } from './criteria.js';

export class DecisionTree {
    #depth = 0;
    #maxDepth = 5;
    #criterion = 'gini';
    #threshold = null;
    #label = null;
    #left = null;
    #right = null;
    #splitAttribute = null;

    constructor({ maxDepth = 5, criterion = 'gini' }) {
        this.#maxDepth = maxDepth;
        this.#criterion = criterion;
    }

    fit(x, y) {
        const impurity = this.#criterion === 'gini' ? gini(y) : entropy(y);

        // When the impurity is 0 or the max depth is reached, we have reached a leaf node
        if (this.#depth >= this.#maxDepth || impurity === 0) {
            this.#label = this.#majorityVote(y);
            return;
        }

        // Find the best split
        const { attribute, threshold } = this.#findBestSplit(x, y);
        this.#splitAttribute = attribute;
        this.#threshold = threshold;

        // Split the data
        const { leftFeatures, leftLabels, rightFeatures, rightLabels } =
            this.#splitData(x, y, attribute, threshold);

        // Build the left and right subtrees
        this.#left = new DecisionTree(
            this.#depth + 1,
            this.#maxDepth,
            this.#criterion
        );
        this.#left.fit(leftFeatures, leftLabels);

        this.#right = new DecisionTree(
            this.#depth + 1,
            this.#maxDepth,
            this.#criterion
        );
        this.#right.fit(rightFeatures, rightLabels);
    }

    #majorityVote(labels) {
        const counts = {};
        labels.forEach((label) => {
            counts[label] = (counts[label] || 0) + 1;
        });
        return Object.keys(counts).reduce((a, b) =>
            counts[a] > counts[b] ? a : b
        );
    }

    // Find the best split for a given set of features and labels
    #findBestSplit(features, labels) {
        let bestImpurity = Infinity;
        let bestAttribute = null;
        let bestThreshold = null;

        // Loop through each attribute
        for (let attribute = 0; attribute < features[0].length; attribute++) {
            const values = features.map((row) => row[attribute]);
            const uniqueValues = Array.from(new Set(values)).sort(
                (a, b) => a - b
            );

            // Find the best threshold for the attribute
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;

                const { leftLabels, rightLabels } =
                    this.#splitLabelsByThreshold(
                        features,
                        labels,
                        attribute,
                        threshold
                    );

                // Calculate the impurity of the split
                const impurity = calculateSplitImpurity(
                    leftLabels,
                    rightLabels,
                    this.#criterion
                );
                if (impurity < bestImpurity) {
                    bestImpurity = impurity;
                    bestAttribute = attribute;
                    bestThreshold = threshold;
                }
            }
        }
        return { attribute: bestAttribute, threshold: bestThreshold };
    }

    #splitLabelsByThreshold(features, labels, attribute, threshold) {
        const leftLabels = [],
            rightLabels = [];
        features.forEach((row, index) => {
            if (row[attribute] < threshold) {
                leftLabels.push(labels[index]);
            } else {
                rightLabels.push(labels[index]);
            }
        });
        return { leftLabels, rightLabels };
    }

    #splitData(features, labels, attribute, threshold) {
        const leftFeatures = [],
            rightFeatures = [];
        const leftLabels = [],
            rightLabels = [];
        features.forEach((row, index) => {
            if (row[attribute] < threshold) {
                leftFeatures.push(row);
                leftLabels.push(labels[index]);
            } else {
                rightFeatures.push(row);
                rightLabels.push(labels[index]);
            }
        });
        return { leftFeatures, leftLabels, rightFeatures, rightLabels };
    }

    predict(features) {
        if (this.#label !== null) {
            return this.#label;
        }
        if (features[this.#splitAttribute] < this.#threshold) {
            return this.#left.predict(features);
        } else {
            return this.#right.predict(features);
        }
    }
}
