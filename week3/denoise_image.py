import cv2
from matplotlib import pyplot as plt


def denoinseImage(image, gt = None):
    gauss = cv2.GaussianBlur(image, (5, 5), 0)
    median = cv2.medianBlur(image, 5)
    bilateral = cv2.bilateralFilter(image, 20, 40, 10)
    denoise_1 = cv2.fastNlMeansDenoisingColored(image,None,3,3,7,21)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    if gt is not None:
        ax1.set_title(f'Input PSNR {cv2.PSNR(gt, image):.3f}')
        ax2.set_title(f'Gauss PSNR {cv2.PSNR(gt, gauss):.3f}')
        ax3.set_title(f'Median PSNR {cv2.PSNR(gt, median):.3f}')
        ax4.set_title(f'Bilateral PSNR {cv2.PSNR(gt, bilateral):.3f}')
        ax5.set_title(f'FastNl Means PSNR {cv2.PSNR(gt, denoise_1):.3f}')
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB))
    ax3.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    ax4.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
    ax5.imshow(cv2.cvtColor(denoise_1, cv2.COLOR_BGR2RGB))
    plt.show()
    