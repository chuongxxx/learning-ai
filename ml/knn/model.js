import * as tf from "@tensorflow/tfjs";

import { euclideanDistance } from "./distance.js";

class BaseKNN {
    constructor({ k = 3, distanceFunction }) {
        this.k = k;
        this.distanceFunction = distanceFunction || euclideanDistance;
    }

    fit(x, y) {
        this.x = tf.tensor2d(x, [x.length, x[0].length]);
        this.y = tf.tensor1d(y);
    }

    getNeighbors(x) {
        const tfX = tf.tensor1d(x);
        const distances = this.distanceFunction(tfX, this.x);

        const { indices: topKIndices } = tf.topk(tf.neg(distances), this.k);

        return tf.gather(this.y, topKIndices);
    }
}

export class KNNClassifier extends BaseKNN {
    predict(x) {
        const predictions = x.map((item) => this.getNeighbors(item));
        return predictions.map((prediction) =>
            this._findMostCommonLabel(prediction)
        );
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

export class KNNRegressor extends BaseKNN {
    predict(x) {
        const predictions = x.map((item) => this.getNeighbors(item));
        return predictions.map((prediction) => tf.mean(prediction).arraySync());
    }
}
