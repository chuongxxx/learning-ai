export class MSE {
    static gradient(prediction, y) {
        return y.map((yi, i) => prediction[i] - yi);
    }

    static hessian(prediction, y) {
        return y.map(() => 1); // ∂^2L/∂f(x)^2 = 1 for MSE
    }

    static loss(prediction, y) {
        return (
            y.reduce((a, b, i) => a + (b - prediction[i]) ** 2, 0) / y.length
        );
    }
}
