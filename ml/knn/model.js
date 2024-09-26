import * as tf from '@tensorflow/tfjs';

import { euclideanDistance } from './distance.js';

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
    topKIndices.print();
    return tf.gather(this.y, topKIndices);
  }
}

export class KNNClassifier extends BaseKNN {
  predict(x) {
    const predictions = x.map((item) => this.getNeighbors(item));
    return predictions.map((prediction) =>
      this._findMostCommonLabel(prediction),
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

// KNN with K-D Tree
// https://en.wikipedia.org/wiki/K-d_tree

class KDTreeNode {
  constructor(point, value, axis) {
    this.point = point;
    this.value = value;
    this.axis = axis;
    this.left = null;
    this.right = null;
  }
}

class KDTree {
  constructor(points, values, depth = 0, distanceFunction) {
    this.distanceFunction = distanceFunction || euclideanDistance;
    const n = points.shape[1];
    const axis = depth % n;

    if (points.shape[0] === 0) {
      this.node = null;
    } else {
      const sortedIndices = points
        .slice([0, axis], [-1, 1])
        .arraySync()
        .map((val, idx) => ({ val: val[0], idx }))
        .sort((a, b) => a.val - b.val)
        .map((el) => el.idx);

      const median = Math.floor(sortedIndices.length / 2);

      const point = points.slice([sortedIndices[median], 0], [1, -1]);
      const value = values.gather([sortedIndices[median]]);

      this.node = new KDTreeNode(point, value, axis);

      this.node.left = new KDTree(
        points.gather(sortedIndices.slice(0, median)),
        values.gather(sortedIndices.slice(0, median)),
        depth + 1,
      );

      this.node.right = new KDTree(
        points.gather(sortedIndices.slice(median + 1)),
        values.gather(sortedIndices.slice(median + 1)),
        depth + 1,
      );
    }
  }

  getNeighbors(targetPoint, k, depth = 0, best = []) {
    if (!this.node) return best;

    const axis = this.node.axis;
    const targetPointData = targetPoint.dataSync();
    const nodePointData = this.node.point.dataSync();
    const distance = this.distanceFunction(
      this.node.point,
      targetPoint,
    ).dataSync();

    best.push({
      point: this.node.point,
      value: this.node.value,
      distance: distance[0],
    });
    best.sort((a, b) => a.distance - b.distance);
    if (best.length > k) best.pop();

    const direction =
      targetPointData[axis] < nodePointData[axis] ? 'left' : 'right';
    const nextBranch = this.node[direction];
    const otherBranch = direction === 'left' ? this.node.right : this.node.left;

    if (nextBranch) {
      best = nextBranch.getNeighbors(targetPoint, k, depth + 1, best);
    }

    if (
      otherBranch &&
      Math.abs(targetPointData[axis] - nodePointData[axis]) <
        best[best.length - 1].distance
    ) {
      best = otherBranch.getNeighbors(targetPoint, k, depth + 1, best);
    }

    return best;
  }
}

class BaseKNNWithKDTree extends BaseKNN {
  fit(x, y) {
    super.fit(x, y);
    this.kdTree = new KDTree(this.x, this.y);
  }
}

export class KNNRegressorWithKDTree extends BaseKNNWithKDTree {
  predict(x) {
    const predictions = x.map((item) => {
      const tfItem = tf.tensor1d(item);
      const neighbors = this.kdTree.getNeighbors(tfItem, this.k, new Array());
      const values = neighbors.map((neighbor) => neighbor.value.arraySync());
      return tf.mean(values).arraySync();
    });

    return predictions;
  }
}

export class KNNWithKDTreeClassifier extends BaseKNNWithKDTree {
  predict(x) {
    const predictions = x.map((item) => {
      const tfItem = tf.tensor1d(item);
      const neighbors = this.kdTree.getNeighbors(tfItem, this.k, new Array());

      const values = neighbors.map((neighbor) => neighbor.value.arraySync());

      const mode = this._mode(values);
      return mode;
    });

    return predictions;
  }

  _mode(arr) {
    const frequencyMap = {};
    let maxFreq = 0;
    let mode = null;

    arr.forEach((value) => {
      frequencyMap[value] = (frequencyMap[value] || 0) + 1;
      if (frequencyMap[value] > maxFreq) {
        maxFreq = frequencyMap[value];
        mode = value;
      }
    });

    return mode;
  }
}

// KNN with Ball Tree
// https://en.wikipedia.org/wiki/Ball_tree
