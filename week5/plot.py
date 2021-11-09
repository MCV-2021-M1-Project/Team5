from matplotlib import pyplot as plt
import os

def plotResults(results, kBest, imagesDDBB, queryImage):
    # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(queryImage)
    plt.axis("off")

    # get method names to a list
    methodNames = []
    for methodName, values in results.items():
        methodNames.append(methodName)

    # initialize the results figure
    fig, big_axes = plt.subplots(nrows=len(methodNames), ncols=1)
    fig.suptitle('')
    fig.tight_layout(h_pad=1.2)

    # set row names
    for row, big_ax in enumerate(big_axes, start=0):
        big_ax.set_title(methodNames[row], fontsize=10, y = 1.3)
        big_ax.axis("off")

    # plot each image in subplot
    for (j, (methodName, values)) in enumerate(results.items()):

        bestKValues = values[0:kBest]

        # loop over the results
        for (i, (v, k)) in enumerate(bestKValues):
            # show the result
            ax = fig.add_subplot(len(methodNames), kBest, j * kBest + i + 1)
            ax.set_title("%s: %.2f" % (os.path.basename(k), v), fontsize = 5)
            plt.imshow(imagesDDBB[k])
            plt.axis("off")

    # show the OpenCV methods
    plt.show()
