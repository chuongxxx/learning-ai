# K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a machine learning algorithm used for both classification and regression. Below are the main steps of the KNN algorithm.

**K-Nearest Neighbors (KNN)** là một thuật toán học máy được sử dụng cho phân loại và hồi quy. Dưới đây là các bước chính của thuật toán KNN.

## I. KNN Classifier

KNN Classifier is used to classify data points based on the majority class among the `k` nearest neighbors.

### 1. Select K-Nearest Neighbors

Choose the number of nearest neighbors `k` to consider during classification.

#### 1. Chọn K-Nearest Neighbors

_Chọn số lượng hàng xóm gần nhất `k` để xem xét trong quá trình phân loại._

### 2. Find the distance between the query point and all training points

Use one of the distance formulas to calculate the distance from the query point to all training points.

#### 2. Tính khoảng cách giữa điểm truy vấn và tất cả các điểm huấn luyện

_Sử dụng một trong các công thức khoảng cách để tính khoảng cách từ điểm truy vấn đến tất cả các điểm huấn luyện._

### 3. Sort the training points by distance

Sort the training points from closest to farthest relative to the query point.

#### 3. Sắp xếp các điểm huấn luyện theo khoảng cách

_Sắp xếp các điểm huấn luyện từ gần đến xa so với điểm truy vấn._

### 4. Select the k closest points

Take the `k` nearest points from the sorted list.

#### 4. Chọn k điểm gần nhất

_Lấy `k` điểm gần nhất từ danh sách đã sắp xếp._

### 5. Find the most common class among the k closest points

Analyze the classes of the `k` nearest points and find the most frequent class.

#### 5. Tìm lớp phổ biến nhất trong số k điểm gần nhất

_Phân tích các lớp của `k` điểm gần nhất và tìm lớp xuất hiện nhiều nhất._

### 6. Return the class with the highest number of votes

Return the class with the highest number of votes from the nearest points.

#### 6. Trả về lớp có số phiếu cao nhất

_Trả về lớp với số lượng phiếu bầu cao nhất từ các điểm gần nhất._

---

## II. KNN Regression

KNN Regression is used to predict continuous values based on the average of the target values of the `k` nearest neighbors.

### 1. Select K-Nearest Neighbors

Choose the number of nearest neighbors `k` to consider during regression.

#### 1. Chọn K-Nearest Neighbors

_Chọn số lượng hàng xóm gần nhất `k` để xem xét trong quá trình hồi quy._

### 2. Find the distance between the query point and all training points

Use one of the distance formulas to calculate the distance from the query point to all training points.

#### 2. Tính khoảng cách giữa điểm truy vấn và tất cả các điểm huấn luyện

_Sử dụng một trong các công thức khoảng cách để tính khoảng cách từ điểm truy vấn đến tất cả các điểm huấn luyện._

### 3. Sort the training points by distance

Sort the training points from closest to farthest relative to the query point.

#### 3. Sắp xếp các điểm huấn luyện theo khoảng cách

_Sắp xếp các điểm huấn luyện từ gần đến xa so với điểm truy vấn._

### 4. Select the k closest points

Take the `k` nearest points from the sorted list.

#### 4. Chọn k điểm gần nhất

_Lấy `k` điểm gần nhất từ danh sách đã sắp xếp._

### 5. Compute the average target value among the k closest points

For regression, calculate the average of the target values of the `k` nearest neighbors.

#### 5. Tính giá trị trung bình của các giá trị đích trong số k điểm gần nhất

_Đối với hồi quy, tính giá trị trung bình của các giá trị đích của `k` điểm gần nhất._

### 6. Return the predicted value

Return the average as the predicted value.

#### 6. Trả về giá trị dự đoán

_Trả về giá trị trung bình như là giá trị dự đoán._

---

## III. Some common distances

Here are some common distance measures that can be used in KNN:

-   **Euclidean distance**: [Learn more](https://en.wikipedia.org/wiki/Euclidean_distance)
-   **Manhattan distance**: [Learn more](https://en.wikipedia.org/wiki/Manhattan_distance)
-   **Chebyshev distance**: [Learn more](https://en.wikipedia.org/wiki/Chebyshev_distance)
-   **Hamming distance**: [Learn more](https://en.wikipedia.org/wiki/Hamming_distance)
-   **Minkowski distance**: [Learn more](https://en.wikipedia.org/wiki/Minkowski_distance)

### III. Một số khoảng cách phổ biến

Dưới đây là một số khoảng cách phổ biến có thể sử dụng trong KNN:

-   **Khoảng cách Euclidean**: [Tìm hiểu thêm](https://en.wikipedia.org/wiki/Euclidean_distance)
-   **Khoảng cách Manhattan**: [Tìm hiểu thêm](https://en.wikipedia.org/wiki/Manhattan_distance)
-   **Khoảng cách Chebyshev**: [Tìm hiểu thêm](https://en.wikipedia.org/wiki/Chebyshev_distance)
-   **Khoảng cách Hamming**: [Tìm hiểu thêm](https://en.wikipedia.org/wiki/Hamming_distance)
-   **Khoảng cách Minkowski**: [Tìm hiểu thêm](https://en.wikipedia.org/wiki/Minkowski_distance)
