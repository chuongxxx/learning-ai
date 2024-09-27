# Decision Tree Construction Process

## 1. Introduction

A **Decision Tree** is a supervised learning algorithm used for both classification and regression tasks. The tree splits data based on feature values to make predictions. The goal is to construct a tree that best separates the data by minimizing impurity at each split, thus improving the model's accuracy in classification or minimizing error in regression.

### Key Concepts:

-   **Node**: A point in the tree where data is split.
-   **Leaf Node**: The terminal node where a final decision or prediction is made.
-   **Root Node**: The top node representing the entire dataset.
-   **Splitting**: The process of dividing the dataset into smaller subsets based on feature values.
-   **Depth**: The number of levels from the root to a leaf node.
-   **Criterion**: The measure used to evaluate the quality of a split (e.g., Gini Index, Entropy, Mean Squared Error).

---

## 2. Decision Tree Types

### 2.1 Decision Tree Classifier

A **Decision Tree Classifier** is used for classification tasks where the target variable is categorical. The goal is to split the data into subsets that are as homogeneous as possible regarding the target class.

-   **Criterion for Splitting**: Measures such as **Gini Index** or **Entropy** are used to evaluate the quality of splits.
-   **Prediction**: At the leaf nodes, the majority class of the samples in that node is used as the prediction.

### 2.2 Decision Tree Regressor

A **Decision Tree Regressor** is used for regression tasks where the target variable is continuous. The goal is to split the data in a way that minimizes the variance in the target variable for each subset.

-   **Criterion for Splitting**: The **Mean Squared Error (MSE)** is commonly used to evaluate splits.
-   **Prediction**: At the leaf nodes, the mean value of the target variable in that node is used as the prediction.

---

## 3. Criteria for Splitting

### 3.1 Gini Impurity (For Classification)

The **Gini Index** measures the probability of incorrectly classifying a randomly chosen sample from a subset. A lower Gini Index indicates a more homogeneous subset, meaning the samples belong to a single class more often.

#### Formula:

$$ Gini = 1 - \sum\_{i=1}^{n} p_i^2 $$
Where $p_i$ is the proportion of samples that belong to class $i$.

#### Example:

For a subset with 50% Class A and 50% Class B:
$$ Gini = 1 - (0.5^2 + 0.5^2) = 0.5 $$

### 3.2 Entropy (For Classification)

**Entropy** is a measure of the uncertainty or disorder in a dataset. The goal is to reduce entropy, making the subsets as pure (homogeneous) as possible.

#### Formula:

$$ Entropy = - \sum\_{i=1}^{n} p_i \log_2 p_i $$
Where $p_i$ is the proportion of samples that belong to class $i$.

#### Example:

For a subset with 50% Class A and 50% Class B:
$$ Entropy = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1 $$

### 3.3 Mean Squared Error (For Regression)

In regression tasks, **Mean Squared Error (MSE)** is used to evaluate the quality of a split. MSE measures the average of the squared differences between actual and predicted values.

#### Formula:

$$ MSE = \frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y})^2 $$
Where $y_i$ is the actual value, and $\hat{y}$ is the predicted value (mean of the subset).

#### Example:

For actual values [3, 4, 5] and predicted value 4:
$$ MSE = \frac{1}{3}((3 - 4)^2 + (4 - 4)^2 + (5 - 4)^2) = 0.67 $$

---

## 4. Decision Tree Construction Steps

### 4.1 Initial Setup

Initialize the Decision Tree with parameters such as the maximum depth and the criterion for splitting.

### 4.2 Calculate Impurity or Error

At each node, calculate the impurity (for classification) or error (for regression) using the specified criterion. If the impurity is 0 (or error is minimal), or the maximum depth is reached, the node becomes a leaf.

### 4.3 Finding the Best Split

For each feature, evaluate all possible split points and find the threshold that minimizes the impurity or error after splitting.

### 4.4 Split the Data

Split the dataset into two subsets based on the selected feature and threshold:

-   Left Subset: Samples where the feature value is less than the threshold.
-   Right Subset: Samples where the feature value is greater than or equal to the threshold.

### 4.5 Recursively Build Subtrees

Recursively apply the same procedure to the left and right subsets until leaf nodes are reached or the maximum depth is attained.

### 4.6 Leaf Nodes and Predictions

At the leaf nodes, the tree either returns the majority class (for classification) or the mean value (for regression) as the prediction.

## 5. Decision Tree Classifier vs. Decision Tree Regressor

The following table summarizes the key differences between Decision Tree Classifiers and Regressors:

| **Aspect**                  | **Classifier**                                                 | **Regressor**                                    |
| --------------------------- | -------------------------------------------------------------- | ------------------------------------------------ |
| **Target Variable**         | Categorical                                                    | Continuous                                       |
| **Splitting Criteria**      | Gini Impurity / Entropy                                        | Mean Squared Error (MSE)                         |
| **Prediction at Leaf Node** | Majority class of the data subset                              | Mean value of the data subset                    |
| **Use Cases**               | Classification tasks (e.g., spam detection, image recognition) | Regression tasks (e.g., predicting house prices) |
