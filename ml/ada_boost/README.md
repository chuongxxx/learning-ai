# Summary of AdaBoost / Tóm tắt về AdaBoost

## Introduction / Giới thiệu

**AdaBoost (Adaptive Boosting)** is a machine learning algorithm used to enhance the performance of weak classifiers by combining them into a strong classifier.  
_**AdaBoost (Adaptive Boosting)** là một thuật toán học máy được sử dụng để cải thiện hiệu suất của các mô hình phân loại yếu (weak classifiers) bằng cách kết hợp chúng thành một mô hình phân loại mạnh (strong classifier)._

## Key Concepts / Các khái niệm chính

-   **Weak Classifier**: A classification model that performs better than random guessing but is not strong enough to stand alone.  
    _**Weak Classifier**: Là một mô hình phân loại có hiệu suất tốt hơn ngẫu nhiên, nhưng không đủ mạnh để hoạt động độc lập._

-   **Strong Classifier**: A classification model created by combining multiple weak classifiers.  
    _**Strong Classifier**: Là một mô hình phân loại được tạo ra từ việc kết hợp nhiều weak classifiers._

-   **Weighted Voting**: Each weak classifier is assigned a weight reflecting its accuracy. This weight is used to calculate the final prediction.  
    _**Weighted Voting**: Mỗi weak classifier được gán một trọng số, phản ánh độ chính xác của nó. Trọng số này được sử dụng để tính toán dự đoán cuối cùng._

## Steps of AdaBoost / Các bước thực hiện của AdaBoost

### Step 1: Initialize Weights / Bước 1: Khởi tạo trọng số

-   Assign equal weights to all samples in the training set. If there are $N$ samples, the initial weight for each is  
    _Gán trọng số bằng nhau cho tất cả các mẫu trong tập huấn luyện. Nếu có $N$ mẫu, mỗi trọng số ban đầu là_

$$
w_i = \frac{1}{N}, \quad i = 1, 2, \ldots, N
$$

### Step 2: Train Weak Classifier (Stump) / Bước 2: Huấn luyện Weak Classifier (Stump)

-   Train a weak classifier on the training set with updated weights. Algorithms like Decision Tree with limited depth (e.g., a depth-1 decision tree) can be used.  
    _Huấn luyện một weak classifier trên tập huấn luyện với trọng số đã được cập nhật. Có thể sử dụng các thuật toán như Decision Tree với chiều sâu giới hạn (như cây quyết định có chiều sâu 1)._

### Step 3: Calculate Error / Bước 3: Tính toán lỗi

-   Calculate the error rate of the weak classifier. The error rate is the total weight of the misclassified samples:  
    _Tính toán tỷ lệ lỗi của weak classifier. Tỷ lệ lỗi là tổng trọng số của các mẫu bị phân loại sai:_

$$
\text{Error} = \frac{\sum_{i=1}^N w_i \cdot I(y_i \neq h(x_i))}{\sum_{i=1}^N w_i}
$$

Where:  
Trong đó:

-   $w_i$ is the weight of sample $i$.  
    $w_i$ là trọng số của mẫu $i$.
-   $I$ is the indicator function, equal to 1 if misclassified and 0 if correct.  
    $I$ là hàm chỉ thị, bằng 1 nếu phân loại sai và 0 nếu đúng.

### Step 4: Update Weights / Bước 4: Cập nhật trọng số

-   Calculate the weight for the weak classifier:  
    _Tính trọng số cho weak classifier:_

$$
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \text{Error}}{\text{Error}}\right)
$$

-   Update weights for the samples:  
    _Cập nhật trọng số cho các mẫu:_

$$
w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i h(x_i))
$$

-   Normalize the weights so that the total weight equals 1:  
    _Normalize các trọng số để tổng trọng số bằng 1:_

$$
w_i = \frac{w_i}{\sum_{j=1}^N w_j}
$$

### Step 5: Repeat / Bước 5: Lặp lại

-   Repeat steps 2 to 4 until a specified number of weak classifiers is reached or no further improvement is observed.  
    _Lặp lại các bước 2 đến 4 cho đến khi đạt được số lượng weak classifiers nhất định hoặc khi không còn cải thiện nào._

### Step 6: Create Strong Classifier / Bước 6: Tạo Strong Classifier

-   Combine the weak classifiers into a strong classifier:  
    _Kết hợp các weak classifiers thành một strong classifier:_

$$
H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)
$$

Where:  
_Trong đó:_

-   $H(x)$ is the strong classifier.  
    $H(x)$ là strong classifier.
-   $h_t(x)$ is the weak classifier at iteration $t$.  
    $h_t(x)$ là weak classifier tại vòng lặp $t$.
-   $\alpha_t$ is the weight of the weak classifier.  
    $\alpha_t$ là trọng số của weak classifier.

## Conclusion / Kết luận

**AdaBoost** is a powerful and flexible method in machine learning, especially in classification tasks. It works effectively even with simple weak classifiers and can significantly improve the model's accuracy.  
_**AdaBoost** là một phương pháp mạnh mẽ và linh hoạt trong học máy, đặc biệt trong các bài toán phân loại. Nó hoạt động hiệu quả ngay cả với các weak classifiers đơn giản và có thể cải thiện đáng kể độ chính xác của mô hình._

## References / Tài liệu tham khảo

-   [Wikipedia: AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
