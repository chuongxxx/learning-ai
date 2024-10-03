# Gradient Boosting

## Introduction (Giới thiệu)

**Gradient Boosting** is a powerful machine learning algorithm, commonly used for regression and classification problems. It combines multiple weak models (usually decision trees) to form a strong model, with the goal of minimizing prediction error by optimizing the gradient of the loss function.

_**Gradient Boosting** là một thuật toán học máy mạnh mẽ, thường được sử dụng cho các bài toán hồi quy và phân loại. Nó kết hợp nhiều mô hình yếu (thường là cây quyết định) thành một mô hình mạnh, nhằm giảm thiểu lỗi dự đoán bằng cách tối ưu hóa gradient của hàm mất mát._

### Key Concepts (Các khái niệm chính):

-   **Boosting**: A method that combines multiple weak models to create a strong model.
    -   _**Boosting**: Phương pháp kết hợp nhiều mô hình yếu để tạo ra một mô hình mạnh._
-   **Gradient Descent**: An optimization method that minimizes a loss function by moving along the gradient.
    -   _**Gradient Descent**: Phương pháp tối ưu hóa dựa trên gradient để giảm thiểu hàm mất mát._
-   **Residual (Gradient)**: The error between the true value and the predicted value of the model.
    -   _**Residual (Gradient)**: Là phần sai lệch giữa giá trị thực tế và giá trị dự đoán của mô hình._

---

## Steps to Implement Gradient Boosting (Các bước triển khai Gradient Boosting)

### 1. Model Initialization (Khởi tạo mô hình)

Initially, the model's prediction is set to the mean value of the target variable $y$. Given a dataset of points $(X_i, y_i)$, the initial prediction for all values is:

_Ban đầu, dự đoán của mô hình được khởi tạo bằng giá trị trung bình của biến mục tiêu $y$. Giả sử tập dữ liệu gồm các điểm $(X_i, y_i)$, mô hình khởi đầu dự đoán tất cả các giá trị:_

$$\hat{y}_0 = \frac{1}{N} \sum_{i=1}^{N} y_i$$

Where:

-   $\hat{y}_0$: Initial prediction (Dự đoán ban đầu)
-   $y_i$: Actual value (Giá trị thực tế)
-   $N$: Number of data points (Số lượng điểm dữ liệu)

### 2. Loss Function (Hàm mất mát)

Gradient Boosting optimizes based on a loss function. For regression tasks, the most common loss function is **Mean Squared Error (MSE)**, defined as:

Gradient Boosting tối ưu hóa dựa trên hàm mất mát. Đối với bài toán hồi quy, hàm mất mát phổ biến nhất là **Mean Squared Error (MSE)**, được định nghĩa như sau:

$$L(y, \hat{y}) = \frac{1}{2N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2$$

Where:

-   $y_i$: Actual value (Giá trị thực tế)
-   $\hat{y}_i$: Predicted value (Giá trị dự đoán của mô hình)

### 3. Compute Gradient (Residuals) (Tính gradient (phần dư))

The gradient of the loss function for each data point, also known as **residuals**, is the difference between the actual value and the predicted value. This gradient indicates how far off the current model's prediction is from the actual value, and helps adjust the model.

Gradient của hàm mất mát đối với từng điểm dữ liệu, hay còn gọi là **phần dư (residuals)**, là phần sai lệch giữa giá trị thực tế và giá trị dự đoán. Gradient này cho biết mô hình hiện tại dự đoán lệch bao nhiêu so với thực tế, và cần được điều chỉnh để giảm lỗi.

$$r_i = y_i - \hat{y}_i$$

Where:

-   $r_i$: Gradient (Residual) (Gradient (Phần dư))
-   $y_i$: Actual value (Giá trị thực tế)
-   $\hat{y}_i$: Current prediction (Dự đoán hiện tại)

### 4. Train a Weak Model (Huấn luyện mô hình yếu)

At each iteration, a weak model (usually a decision tree) is trained on the residuals $r_i$ instead of the actual values $y_i$. The objective is to have this weak model predict the residuals to adjust the overall model’s prediction.

_Ở mỗi vòng lặp, một mô hình yếu (thường là cây quyết định) được huấn luyện trên phần dư $r_i$ thay vì giá trị $y_i$. Mục tiêu là mô hình yếu này sẽ cố gắng dự đoán phần dư để điều chỉnh giá trị dự đoán của mô hình tổng thể._

### 5. Update the Prediction (Cập nhật dự đoán)

After training the new weak model, the overall prediction is updated by adding a small fraction $\alpha$ of the weak model’s prediction. This gradual adjustment helps the model slowly improve its predictions.

Sau khi huấn luyện mô hình yếu mới, giá trị dự đoán của mô hình tổng thể được cập nhật bằng cách cộng thêm một phần nhỏ $\alpha$ của dự đoán từ mô hình yếu. Điều này giúp mô hình dần dần điều chỉnh để cải thiện dự đoán.

$$\hat{y}_{t+1} = \hat{y}_t + \alpha \cdot h_t(X)$$

Where:

-   $\hat{y}_{t+1}$: New prediction after the $t+1$-th iteration (Dự đoán mới sau vòng lặp $t+1$)
-   $\hat{y}_t$: Prediction after the $t$-th iteration (Dự đoán sau vòng lặp thứ $t$)
-   $h_t(X)$: Prediction from the weak model at iteration $t$ (Dự đoán từ mô hình yếu ở vòng lặp thứ $t$)
-   $\alpha$: Learning rate (Tỉ lệ học)

### 6. Repeat (Lặp lại)

This process is repeated for multiple iterations, with each iteration trying to correct the prediction by learning from the residuals. After $T$ iterations, the final prediction is the sum of the initial prediction and the adjustments from all the weak models:

Quá trình này được lặp lại qua nhiều vòng lặp (iterations), với mỗi lần mô hình cố gắng điều chỉnh dự đoán bằng cách học từ phần dư. Sau $T$ vòng lặp, giá trị dự đoán cuối cùng của mô hình là tổng của tất cả các dự đoán từ các mô hình yếu đã được huấn luyện:

$$\hat{y}_T = \hat{y}_0 + \sum_{t=1}^{T} \alpha \cdot h_t(X)$$
