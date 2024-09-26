import * as tf from '@tensorflow/tfjs';

export function matrixInverse(matrix) {
  const [m, n] = matrix.shape;

  if (m !== n) {
    throw new Error('Matrix must be square!');
  }

  // Identity matrix
  const identity = tf.eye(m);

  let augmented = matrix.concat(identity, 1);

  // Gauss-Jordan
  augmented = augmented.arraySync();
  for (let i = 0; i < m; i++) {
    const divisor = augmented[i][i];
    for (let j = 0; j < 2 * m; j++) {
      augmented[i][j] /= divisor;
    }

    for (let k = 0; k < m; k++) {
      if (k !== i) {
        const factor = augmented[k][i];
        for (let j = 0; j < 2 * m; j++) {
          augmented[k][j] -= factor * augmented[i][j];
        }
      }
    }
  }

  // Augmented matrix
  const inverse = augmented.map((row) => row.slice(m, 2 * m));

  return tf.tensor2d(inverse);
}
