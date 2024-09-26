import * as tf from '@tensorflow/tfjs';
import { matrixInverse } from '../utils/matrix.util.js';

//Ordinary Least Squares
export class LinearRegressionOLS {
  constructor() {}

  /**
   * Fit the model to the given data.
   * @param {array} x The feature data. This should be a 2D array where each
   *     row is a sample and each column is a feature.
   * @param {array} y The label data. This should be a 1D array with the same
   *     length as the number of samples in x.
   * @throws If the input arrays are not valid.
   */
  fit(x, y) {
    if (!Array.isArray(x) || !Array.isArray(x[0])) {
      throw new Error('2D array is required for features');
    }
    if (!Array.isArray(y)) {
      throw new Error('Array is required for labels');
    }
    if (x.length !== y.length) {
      throw new Error('Features and Label must have same length!');
    }

    this.X = tf.tensor2d(
      x.map((item) => [1, ...item]),
      [x.length, x[0].length + 1]
    );
    this.y = tf.tensor2d(y, [y.length, 1]);

    const xT = this.X.transpose();
    const xTx = xT.matMul(this.X);

    // w = (X^T * X)^-1 * X^T * y
    this.w = matrixInverse(xTx).matMul(xT).matMul(this.y);
  }

  /**
   * Get the weights of the model.
   * @return {tf.Tensor2D} A 2D tensor with a single row and a single column
   *     for each feature plus one for the bias term.
   */
  getWeights() {
    return this.w;
  }

  /**
   * Predict the output for a given set of features.
   * @param {array} x The feature data. This should be a 1D array.
   * @return {tf.Tensor2D} A 2D tensor with a single row and a single column
   *     for the predicted output.
   * @throws If the input is not a valid array.
   */
  predict(x) {
    if (!Array.isArray(x)) {
      throw new Error('Array is required for features');
    }
    const X = tf.tensor2d([1, ...x], [1, x.length + 1]);
    return X.matMul(this.w);
  }
}

export class LinearRegressionGD1 {
  constructor({ learningRate = 0.001, epochs = 1000 }) {
    this.learningRate = learningRate;
    this.epochs = epochs;
  }

  /**
   * Fit the model to the given data.
   * @param {array} x The feature data. This should be a 2D array where each
   *     subarray represents a single data point.
   * @param {array} y The labels associated with the feature data. This should
   *     be a 1D array of the same length as the feature data.
   * @throws If the input is not a valid array.
   */
  fit(x, y) {
    if (!Array.isArray(x) || !Array.isArray(x[0])) {
      throw new Error('2D array is required for features');
    }
    if (!Array.isArray(y)) {
      throw new Error('Array is required for labels');
    }
    if (x.length !== y.length) {
      throw new Error('Features and Label must have same length!');
    }

    this.X = tf.tensor2d(
      x.map((item) => [1, ...item]),
      [x.length, x[0].length + 1]
    );
    this.y = tf.tensor2d(y, [y.length, 1]);

    // Initialize w
    this.w = tf.ones([this.X.shape[1], 1]);

    for (let i = 0; i < this.epochs; i++) {
      this.gradientDescent();
    }
  }

  /**
   * Gradient descent algorithm for linear regression.
   *
   * @method gradientDescent
   * @private
   */
  gradientDescent() {
    const prediction = this.X.matMul(this.w);
    const error = prediction.sub(this.y);

    const xT = this.X.transpose();

    const gradient = xT.matMul(error).div(this.X.shape[0]);

    // w = w - learningRate * gradient
    this.w = this.w.sub(tf.mul(gradient, this.learningRate));
  }

  /**
   * Get the weights of the model.
   * @return {tf.Tensor2D} A 2D tensor with a single column and a single row
   *     for each feature plus one for the bias term.
   */
  getWeights() {
    return this.w;
  }

  /**
   * Predict the output for a given set of features.
   * @param {array} x The feature data. This should be a 1D array.
   * @return {tf.Tensor2D} A 2D tensor with a single row and a single column
   *     for the predicted output.
   * @throws If the input is not a valid array.
   */
  predict(x) {
    if (!Array.isArray(x)) {
      throw new Error('Array is required for features');
    }

    const X = tf.tensor2d([1, ...x], [1, x.length + 1]);
    return X.matMul(this.w);
  }
}

export class LinearRegressionGD2 {
  /**
   * Constructor for LinearRegressionGD2.
   * @param {Object} [options] - An object containing options for the constructor.
   * @param {number} [options.learningRate=0.001] - The learning rate for the
   *     gradient descent algorithm.
   * @param {number} [options.epochs=20] - The number of epochs to run the
   *     gradient descent algorithm for.
   */
  constructor({ learningRate = 0.001, epochs = 20 }) {
    this.learningRate = learningRate;
    this.epochs = epochs;
  }

  /**
   * Fit the model to the given data.
   * @param {array} x The feature data. This should be a 2D array where each
   *     subarray represents a single data point.
   * @param {array} y The labels associated with the feature data. This should
   *     be a 1D array of the same length as the feature data.
   * @throws If the input is not a valid array.
   */
  fit(x, y) {
    if (!Array.isArray(x) || !Array.isArray(x[0])) {
      throw new Error('2D array is required for features');
    }
    if (!Array.isArray(y)) {
      throw new Error('Array is required for labels');
    }
    if (x.length !== y.length) {
      throw new Error('Features and Label must have same length!');
    }

    this.X = tf.tensor2d(x, [x.length, x[0].length]);
    this.y = tf.tensor2d(y, [y.length, 1]);

    this.w = tf.zeros([this.X.shape[1], 1]);
    this.b = tf.scalar(0);

    for (let i = 0; i < this.epochs; i++) {
      for (let j = 0; j < this.X.shape[0]; j++) {
        const xFeature = this.X.slice([j, 0], [1, this.X.shape[1]]);
        const yLabel = this.y.slice([j, 0], [1, 1]);
        this.gradientDescent(xFeature, yLabel);
      }
    }
  }

  /**
   * Gradient descent algorithm for linear regression.
   *
   * @method gradientDescent
   * @private
   * @param {tf.Tensor2D} xFeature - A 2D tensor with a single row and a single
   *     column for each feature.
   * @param {tf.Tensor2D} yLabel - A 2D tensor with a single row and a single
   *     column for the label.
   */
  gradientDescent(xFeature, yLabel) {
    const prediction = xFeature.matMul(this.w).add(this.b);
    const error = prediction.sub(yLabel);

    const gradientW = xFeature.transpose().matMul(error).mul(this.learningRate);
    const gradientB = error.sum().mul(this.learningRate);

    this.w = this.w.sub(gradientW);
    this.b = this.b.sub(gradientB);
  }

  /**
   * Get the weights of the model.
   * @return {tf.Tensor2D} A 2D tensor with a single column and a single row
   *     for each feature plus one for the bias term.
   */
  getWeights() {
    return tf.concat([this.w, this.b.reshape([1, 1])], 0);
  }
  predict(x) {
    if (!Array.isArray(x)) {
      throw new Error('Array is required for features');
    }

    const X = tf.tensor2d(x, [1, x.length]);
    return X.matMul(this.w).add(this.b);
  }
}

export class LinearRegressionSGD {
  constructor({ learningRate = 0.001, epochs = 1000 }) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.optimizer = tf.train.sgd(this.learningRate); // Use sgd optimizer from TensorFlow.js
  }

  /**
   * Fit the model to the given data.
   * @param {array} x The feature data. This should be a 2D array where each
   *     subarray represents a single data point.
   * @param {array} y The labels associated with the feature data. This should
   *     be a 1D array of the same length as the feature data.
   * @throws If the input is not a valid array.
   */
  fit(x, y) {
    if (!Array.isArray(x) || !Array.isArray(x[0])) {
      throw new Error('2D array is required for features');
    }
    if (!Array.isArray(y)) {
      throw new Error('Array is required for labels');
    }
    if (x.length !== y.length) {
      throw new Error('Features and Label must have same length!');
    }

    this.X = tf.tensor2d(
      x.map((item) => [1, ...item]),
      [x.length, x[0].length + 1]
    );
    this.y = tf.tensor2d(y, [y.length, 1]);

    // Initialize weights using variable and zeros for better control in TensorFlow.js
    this.w = tf.variable(tf.zeros([this.X.shape[1], 1]));

    for (let i = 0; i < this.epochs; i++) {
      this.optimizer.minimize(() => this.loss());
    }
  }

  /**
   * Loss function for linear regression.
   *
   * @method loss
   * @private
   */
  loss() {
    const prediction = this.X.matMul(this.w);
    const error = prediction.sub(this.y);
    const squaredError = error.square().mean();
    return squaredError;
  }

  /**
   * Get the weights of the model.
   * @return {tf.Tensor2D} A 2D tensor with a single column and a single row
   *     for each feature plus one for the bias term.
   */
  getWeights() {
    return this.w;
  }

  /**
   * Predict the output for a given set of features.
   * @param {array} x The feature data. This should be a 1D array.
   * @return {tf.Tensor2D} A 2D tensor with a single row and a single column
   *     for the predicted output.
   * @throws If the input is not a valid array.
   */
  predict(x) {
    if (!Array.isArray(x)) {
      throw new Error('Array is required for features');
    }

    const X = tf.tensor2d([1, ...x], [1, x.length + 1]);
    return X.matMul(this.w);
  }
}

export class LinearRegressionNN {
  /**
   * Constructor for LinearRegressionNN.
   * @param {Object} [options] - An object containing options for the constructor.
   * @param {number} [options.learningRate=0.001] - The learning rate for the
   *     gradient descent algorithm.
   * @param {number} [options.epochs=20] - The number of epochs to run the
   *     gradient descent algorithm for.
   */
  constructor({ learningRate = 0.001, epochs = 20 }) {
    this.epochs = epochs;
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.model.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.sgd(learningRate)
    });
  }

  /**
   * Fit the model to the given data.
   * @param {array} x The feature data. This should be a 2D array where each
   *     subarray represents a single data point.
   * @param {array} y The labels associated with the feature data. This should
   *     be a 1D array of the same length as the feature data.
   * @throws If the input is not a valid array.
   */
  async fit(x, y) {
    const xTrain = tf.tensor2d(x, [x.length, x[0].length]);
    const yTrain = tf.tensor2d(y, [y.length, 1]);
    await this.model.fit(xTrain, yTrain, {
      epochs: this.epochs
      // callbacks: {
      //   onEpochEnd: async (epoch, logs) => {
      //     // console.log("Epoch:", epoch, "--- Loss: ", logs.loss);
      //   },
      // },
    });
  }

  /**
   * Predict the output for a given set of features.
   * @param {array} x The feature data. This should be a 1D array.
   * @return {tf.Tensor2D} A 2D tensor with a single row and a single column
   *     for the predicted output.
   * @throws If the input is not a valid array.
   */
  predict(x) {
    return this.model.predict(tf.tensor2d(x, [x.length, 1]));
  }
}
