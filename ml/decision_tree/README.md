# Decision Tree Construction Process

## 1. Introduction

A **Decision Tree** is a supervised learning algorithm used for classification and regression tasks. The tree splits the data based on feature values to reach a decision about a target variable. The goal of building a Decision Tree is to find the best splits in the dataset that minimize impurity, leading to better classification or regression performance.

### Key Concepts:

-   **Node**: A point in the decision tree where data is split.
-   **Leaf Node**: The end node where a decision or classification is made.
-   **Root Node**: The starting node containing the full dataset.
-   **Splitting**: The process of dividing data into subsets based on a specific feature.
-   **Depth**: The number of levels from the root node to a leaf node.
-   **Criterion**: The method used to evaluate the quality of a split.

---

## 2. Criteria for Splitting

When constructing a Decision Tree, we need to decide how to split the data at each node. Commonly used criteria include **Gini Index** and **Entropy** (Information Gain). These measures help to identify the best splits by reducing the impurity in the data.

### 2.1 Gini Impurity

The **Gini Index** measures the probability of incorrectly classifying a randomly chosen element from the dataset if it were randomly labeled according to the distribution of class labels in the subset. A lower Gini Index indicates a more homogeneous subset.

**Formula**:
$$Gini = 1 - \sum_{i=1}^{n} p_{i}^{2}$$
Where $p_{i}$ is the probability of an element belonging to class $i$.

#### Example:

If the class distribution in a subset is 50% Class A and 50% Class B, the Gini Index is calculated as:

$$
Gini = 1 - (0.5^2 + 0.5^2) = 0.5
$$

### 2.2 Entropy (Information Gain)

**Entropy** measures the disorder or uncertainty in a dataset. The goal is to minimize entropy, meaning creating subsets that are as pure (homogeneous) as possible.

**Formula**:
$$Entropy = - \sum_{i=1}^{n} p_i \log_2 p_i$$
Where $p_i$ is the proportion of class $i$ in the subset.

#### Example:

For a class distribution of 50% Class A and 50% Class B, the Entropy is:
$$Entropy = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1$$

---

## 3. Decision Tree Construction Steps

### 3.1 Initial Setup

Start by initializing the tree with parameters such as:

-   **maxDepth**: Maximum depth of the tree to avoid overfitting.
-   **criterion**: The criterion used to evaluate splits (Gini or Entropy).

```js
const tree = new DecisionTree({ maxDepth: 5, criterion: 'gini' });
```

### 3.2 Calculate Impurity

At each node, calculate the impurity of the current dataset using either Gini or Entropy. This is the measure used to determine how "pure" the subset of data is.

```js
const impurity = this.#criterion === 'gini' ? gini(y) : entropy(y);
```

-   If impurity = 0 (i.e., all instances have the same class), the node becomes a leaf node.
-   If the maximum depth is reached, the node also becomes a leaf node.

### 3.3 Finding the Best Split

To create a split, loop through each feature and find the threshold that minimizes the impurity of the resulting subsets.

-   Loop through attributes: For each attribute, calculate possible split points.
-   Find optimal threshold: Find the threshold that minimizes the impurity after splitting.

```js
const { attribute, threshold } = this.#findBestSplit(x, y);
```

### 3.4 Split the Data

After finding the best attribute and threshold, split the dataset into two parts:

-   Left Subtree: Instances where the feature value is less than the threshold.
-   Right Subtree: Instances where the feature value is greater than or equal to the threshold.

```js
const { leftFeatures, leftLabels, rightFeatures, rightLabels } =
    this.#splitData(x, y, attribute, threshold);
```

### 3.5 Recursively Build Subtrees

Once the data is split, recursively apply the same procedure to the left and right subsets to continue building the tree.

```js
this.#left = new DecisionTree(this.#depth + 1, this.#maxDepth, this.#criterion);
this.#left.fit(leftFeatures, leftLabels);

this.#right = new DecisionTree(
    this.#depth + 1,
    this.#maxDepth,
    this.#criterion
);
this.#right.fit(rightFeatures, rightLabels);
```

### 3.6 Leaf Nodes and Predictions

When a node is a leaf, it stores the majority class of its instances. Predictions for new data are made by traversing the tree from the root to a leaf node, comparing feature values with split thresholds.

```js
if (this.#label !== null) {
    return this.#label;
}

if (features[this.#splitAttribute] < this.#threshold) {
    return this.#left.predict(features);
} else {
    return this.#right.predict(features);
}
```
