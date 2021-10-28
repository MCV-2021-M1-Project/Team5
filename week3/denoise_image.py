import cv2
from matplotlib import pyplot as plt


def denoinseImage(image, gt = None):
    gtNoise = cv2.Laplacian(gt, cv2.CV_64F).var()
    inputsNoise =cv2.Laplacian(image, cv2.CV_64F).var()
    if inputsNoise > 2000.0:
        gauss = cv2.GaussianBlur(image, (5, 5), 0)
        median = cv2.medianBlur(image, 5)
        bilateral = cv2.bilateralFilter(image, 20, 40, 10)
        gaussNoise =cv2.Laplacian(gauss, cv2.CV_64F).var()
        medianNoise =cv2.Laplacian(median, cv2.CV_64F).var()
        bilateralNoise = cv2.Laplacian(bilateral, cv2.CV_64F).var()
        # print(f'Noise detection on gt: {gtNoise}')
        # print(f'Noise detection on input: {inputsNoise}')
        # print(f'Noise detection on gauss: {gaussNoise}')
        # print(f'Noise detection on median: {medianNoise}')
        # print(f'Noise detection on bilateral: {bilateralNoise}')
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # if gt is not None:
        #     ax1.set_title(f'Input PSNR {cv2.PSNR(gt, image):.3f}')
        #     ax2.set_title(f'Gauss PSNR {cv2.PSNR(gt, gauss):.3f}')
        #     ax3.set_title(f'Median PSNR {cv2.PSNR(gt, median):.3f}')
        #     ax4.set_title(f'Bilateral PSNR {cv2.PSNR(gt, bilateral):.3f}')
        # ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # ax2.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB))
        # ax3.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
        # ax4.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
        # plt.show()
        return median if medianNoise > gaussNoise else gauss
    else:
        return image
    