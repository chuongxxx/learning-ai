// 1. Select K-Nearest Neighbors
// 2. Find the distance between the query point and all training points
// 3. Sort the training points by distance
// 4. Select the k closest points
// 5. Find the most common class among the k closest points
// 6. Return the class with the highest number of votes

// Some distances:
// Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
// Manhattan distance: https://en.wikipedia.org/wiki/Manhattan_distance
// Chebyshev distance: https://en.wikipedia.org/wiki/Chebyshev_distance
// Hamming distance: https://en.wikipedia.org/wiki/Hamming_distance
// Minkowski distance: https://en.wikipedia.org/wiki/Minkowski_distance

import * as tf from "@tensorflow/tfjs";

import { findMostCommonLabel } from "../utils/common.util.js";
import { euclideanDistance } from "./distance.js";

export class KNN {
    constructor({ k = 3, distanceFunction }) {
        this.k = k;
        this.distanceFunction = distanceFunction || euclideanDistance;
    }

    fit(x, y) {
        this.x = tf.tensor2d(x, [x.length, x[0].length]);
        this.y = tf.tensor1d(y, "int32");
    }

    predict(x) {
        const predictions = x.map((item) => this._predictOne(item));
        return predictions;
    }

    _predictOne(x) {
        const tfX = tf.tensor1d(x);

        // calculate the distance between the query point and all training points
        const distances = this.distanceFunction(tfX, this.x);

        // Get the k nearest neighbors
        const { indices: topKIndices } = tf.topk(
            tf.neg(distances),
            this.k,
            true
        );

        const nearestLabels = tf.gather(this.y, topKIndices);

        const prediction = findMostCommonLabel(nearestLabels);

        return prediction;
    }

    _findMostCommonLabel(labels) {
        const uniqueLabels = tf.unique(labels);

        let maxArg = undefined;
        uniqueLabels.values.arraySync().forEach((label, index) => {
            const count = labels
                .arraySync()
                .filter((value) => value === label).length;
            if (!maxArg) {
                maxArg = { index, count };
            }
            if (count > maxArg.count) {
                maxArg = { index, count };
            }
        });

        return uniqueLabels.values.arraySync()[maxArg.index];
    }
}
