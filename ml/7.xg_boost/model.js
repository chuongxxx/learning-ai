class InformationGainDecisionTree {
    constructor(maxDepth, minGain = 0.01) {
        this.maxDepth = maxDepth;
        this.minGain = minGain;
        this.root = null;
    }

    fit(X, y) {
        const { gradients, hessians } = this.computeGradientsAndHessians(
            y,
            this.predictInitialValue(y)
        );
        this.root = this.buildTree(X, gradients, hessians, 0);
    }

    // Tính toán gradients và hessians
    computeGradientsAndHessians(y, preds) {
        const gradients = y.map((yi, i) => preds[i] - yi); // ∂L/∂f(x)
        const hessians = gradients.map(() => 1); // ∂^2L/∂f(x)^2 = 1 cho MSE
        return { gradients, hessians };
    }

    // Dự đoán giá trị khởi tạo
    predictInitialValue(y) {
        return y.reduce((a, b) => a + b, 0) / y.length; // Giá trị trung bình
    }

    buildTree(X, gradients, hessians, depth) {
        if (depth >= this.maxDepth || this.shouldPrune(gradients, hessians)) {
            return this.createLeafNode(gradients, hessians);
        }

        // Chia nhánh và tạo các node
        const {
            leftData,
            rightData,
            leftGradients,
            rightGradients,
            leftHessians,
            rightHessians
        } = this.splitNode(X, gradients, hessians);

        const leftNode = this.buildTree(
            leftData,
            leftGradients,
            leftHessians,
            depth + 1
        );
        const rightNode = this.buildTree(
            rightData,
            rightGradients,
            rightHessians,
            depth + 1
        );

        return {
            left: leftNode,
            right: rightNode,
            feature: this.splitFeature,
            threshold: this.splitThreshold
        }; // Tạo node cho cây
    }

    // Kiểm tra điều kiện cắt tỉa
    shouldPrune(gradients, hessians) {
        const gain = this.calculateGain(gradients, hessians);
        return gain < this.minGain;
    }

    calculateGain(gradients, hessians) {
        const totalGradient = gradients.reduce((a, b) => a + b, 0);
        const totalHessian = hessians.reduce((a, b) => a + b, 0);
        return (totalGradient * totalGradient) / (totalHessian + 1e-10);
    }

    // Tạo node lá
    createLeafNode(gradients, hessians) {
        const value =
            gradients.reduce((a, b) => a + b, 0) /
            (hessians.reduce((a, b) => a + b, 0) + 1e-10);
        return { isLeaf: true, value }; // Giá trị cho node lá
    }

    splitNode(X, gradients, hessians) {
        // Thực hiện phân chia đơn giản dựa trên một đặc trưng và ngưỡng
        let bestGain = -Infinity;
        let bestSplit = null;

        // Thử tất cả các đặc trưng và ngưỡng để tìm phân chia tốt nhất
        for (let featureIndex = 0; featureIndex < X[0].length; featureIndex++) {
            const thresholds = [...new Set(X.map((row) => row[featureIndex]))]; // Các ngưỡng khác nhau cho đặc trưng này

            for (let threshold of thresholds) {
                const {
                    leftData,
                    rightData,
                    leftGradients,
                    rightGradients,
                    leftHessians,
                    rightHessians
                } = this.splitData(
                    X,
                    gradients,
                    hessians,
                    featureIndex,
                    threshold
                );

                const gain =
                    this.calculateGain(leftGradients, leftHessians) +
                    this.calculateGain(rightGradients, rightHessians) -
                    this.calculateGain(gradients, hessians);

                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = {
                        featureIndex,
                        threshold,
                        leftData,
                        rightData,
                        leftGradients,
                        rightGradients,
                        leftHessians,
                        rightHessians
                    };
                }
            }
        }

        if (bestSplit) {
            this.splitFeature = bestSplit.featureIndex;
            this.splitThreshold = bestSplit.threshold;
            return {
                leftData: bestSplit.leftData,
                rightData: bestSplit.rightData,
                leftGradients: bestSplit.leftGradients,
                rightGradients: bestSplit.rightGradients,
                leftHessians: bestSplit.leftHessians,
                rightHessians: bestSplit.rightHessians
            };
        }

        return {
            leftData: X,
            rightData: [],
            leftGradients: gradients,
            rightGradients: [],
            leftHessians: hessians,
            rightHessians: []
        };
    }

    splitData(X, gradients, hessians, featureIndex, threshold) {
        const leftData = [];
        const rightData = [];
        const leftGradients = [];
        const rightGradients = [];
        const leftHessians = [];
        const rightHessians = [];

        for (let i = 0; i < X.length; i++) {
            if (X[i][featureIndex] <= threshold) {
                leftData.push(X[i]);
                leftGradients.push(gradients[i]);
                leftHessians.push(hessians[i]);
            } else {
                rightData.push(X[i]);
                rightGradients.push(gradients[i]);
                rightHessians.push(hessians[i]);
            }
        }

        return {
            leftData,
            rightData,
            leftGradients,
            rightGradients,
            leftHessians,
            rightHessians
        };
    }

    predict(x) {
        return this.predictNode(this.root, x);
    }

    predictNode(node, x) {
        if (node.isLeaf) {
            return node.value;
        }

        if (x[node.feature] <= node.threshold) {
            return this.predictNode(node.left, x);
        } else {
            return this.predictNode(node.right, x);
        }
    }
}

export class XGBoost {
    constructor(maxDepth = 3, learningRate = 0.1, nEstimators = 100) {
        this.maxDepth = maxDepth; // Độ sâu tối đa của cây
        this.learningRate = learningRate; // Tốc độ học
        this.nEstimators = nEstimators; // Số lượng cây
        this.baseLearners = []; // Danh sách các cây quyết định
        this.initValue = 0; // Giá trị khởi tạo f(0)(x)
    }

    // Bước 1: Khởi tạo mô hình với giá trị không đổi
    initialize(y) {
        this.initValue = y.reduce((a, b) => a + b, 0) / y.length; // Giá trị trung bình
    }

    // Bước 2: Tính toán gradients và hessians (MSE loss là ví dụ)
    computeGradientsAndHessians(y, preds) {
        const gradients = y.map((yi, i) => preds[i] - yi); // ∂L/∂f(x)
        const hessians = gradients.map(() => 1); // ∂²L/∂f(x)² = 1 cho MSE
        return { gradients, hessians };
    }

    fitWeakLearner(X, gradients) {
        const tree = new InformationGainDecisionTree(this.maxDepth);
        tree.fit(X, gradients); // Huấn luyện cây với gradients
        return tree;
    }

    // Bước 4: Cập nhật mô hình với các cây yếu mới
    updateModel(tree, X) {
        const predictions = X.map((x) => tree.predict(x)); // Dự đoán bằng cây
        return predictions.map((pred) => this.learningRate * pred); // Cập nhật dự đoán với learning rate
    }

    fit(X, y) {
        this.initialize(y); // Khởi tạo mô hình

        let currentPredictions = new Array(y.length).fill(this.initValue); // Bắt đầu với f(0)(x)

        for (let m = 0; m < this.nEstimators; m++) {
            const { gradients, hessians } = this.computeGradientsAndHessians(
                y,
                currentPredictions
            );

            // Huấn luyện cây yếu
            const tree = this.fitWeakLearner(X, gradients, hessians);
            this.baseLearners.push(tree); // Lưu cây vào danh sách

            // Cập nhật dự đoán với cây yếu mới
            const updates = this.updateModel(tree, X);

            // f(m)(x) = f(m-1)(x) + α * new_predictions
            currentPredictions = currentPredictions.map(
                (pred, i) => pred - updates[i]
            );
        }
    }

    predict(X) {
        // Bắt đầu với giá trị khởi tạo
        let predictions = new Array(X.length).fill(this.initValue);

        // Tính tổng tất cả các dự đoán của các cây yếu
        for (let tree of this.baseLearners) {
            const updates = X.map((x) => tree.predict(x)); // Dự đoán từ mỗi cây
            predictions = predictions.map(
                (pred, i) => pred - this.learningRate * updates[i]
            ); // Cập nhật dự đoán
        }

        return predictions; // Trả về dự đoán cuối cùng
    }
}
