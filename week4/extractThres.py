from matplotlib import pyplot as plt
import numpy as np

matches = [224, 133, 67, 19, 24, 278, 42, 140, 66, 154, 109, 43, 223, 169, 51, 211, 195, 0, 42, 23, 168, 212, 18, 8, 29, 168, 138, 34, 144, 249, 171, 263, 16, 15, 23, 223, 221, 433]

answers = [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

minMatch = min(matches) + 1
maxMatches = max(matches)

bestF1 = 0
bestThres = 0

precisions, recalls, f1s = [], [], []

for thres in range(minMatch, maxMatches):
    possibleResult =  []
    for match in matches:
        possibleResult.append(1 if match > thres else 0)
    
    truePositive = np.count_nonzero(np.multiply(answers, possibleResult))
    falseNegative = np.count_nonzero(np.multiply(answers, (1 - np.asarray(possibleResult))))
    falsePositive = np.count_nonzero(np.multiply((1 - np.asarray(answers)), possibleResult))

    if truePositive + falsePositive != 0:
        precision = truePositive / (truePositive + falsePositive)
        recall = truePositive / (truePositive + falseNegative)
    else:
        precision = 0
        recall = 0
    if precision + recall != 0:
        F1_measure = 2 * ((precision * recall) / (precision + recall))
    else:
        F1_measure = 0
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(F1_measure)
    if F1_measure > bestF1:
        bestF1 = F1_measure
        bestThres = thres

print(f'Best F1-socre {bestF1} with thres {bestThres}')
fig, ax = plt.subplots()
ax.set_title('F1-mesure Curve')
ax.set_xlabel('Matches Threshold')
ax.set_ylabel('F1-score')
ax.plot(range(minMatch, maxMatches),f1s)
plt.show()