export const entropy = (labels) => {
    const labelCounts = {};
    labels.forEach((label) => {
        labelCounts[label] = (labelCounts[label] || 0) + 1;
    });

    let entropy = 0;
    const total = labels.length;
    for (let label in labelCounts) {
        const probability = labelCounts[label] / total;
        entropy -= probability * Math.log2(probability);
    }
    return entropy;
};

export const gini = (labels) => {
    const labelCounts = {};
    labels.forEach((label) => {
        labelCounts[label] = (labelCounts[label] || 0) + 1;
    });

    let gini = 1;
    const total = labels.length;
    for (let label in labelCounts) {
        const probability = labelCounts[label] / total;
        gini -= Math.pow(probability, 2);
    }
    return gini;
};

export const calculateSplitImpurity = (leftLabels, rightLabels, criterion) => {
    const total = leftLabels.length + rightLabels.length;
    const leftImpurity =
        criterion === 'gini' ? gini(leftLabels) : entropy(leftLabels);
    const rightImpurity =
        criterion === 'gini' ? gini(rightLabels) : entropy(rightLabels);

    const weightedImpurity =
        (leftLabels.length / total) * leftImpurity +
        (rightLabels.length / total) * rightImpurity;
    return weightedImpurity;
};
