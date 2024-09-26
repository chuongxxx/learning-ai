import * as tf from '@tensorflow/tfjs';

export const euclideanDistance = (x, y) => {
    const diff = tf.sub(x, y);
    const square = tf.square(diff);
    const sum = tf.sum(square, 1);
    return tf.sqrt(sum);
};
