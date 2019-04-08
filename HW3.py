import numpy as np
import cv2
import matplotlib.pyplot as plt

# Padding function
def pad(img):
    M = len(img)
    N = len(img[0])
    P = 2*M
    Q = 2*N

    padded = np.zeros((P,Q),dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            padded[i][j] = img[i][j]

    return padded

# Centering/Decentering function
def center(img):
    M = len(img)
    N = len(img[0])
    centeredImg = img

    for i in range(M):
        for j in range(N):
            centeredImg[i][j] *= (-1)**(i+j)
    return centeredImg

# Multiply by Laplacian of Gaussian filter function
def LoG(img, sigma):
    M = len(img)
    N = len(img[0])
    yOffset = M // 2
    xOffset = N // 2

    LoG = np.zeros((M,N), dtype=np.complex128)
    K = 0
    for i in range(M):
        for j in range(N):
            LoG[i][j] = -1/(np.pi*sigma**4)*(1 - ((j-xOffset)**2 + (i-yOffset)**2)/(2*sigma**2))*np.exp(-((j-xOffset)**2 + (i-yOffset)**2)/(2*sigma**2))
            K += LoG[i][j]
    LoG /= K

    plt.imshow(abs(LoG))
    plt.title('Magnitude Spectrum of LoG Filter')
    plt.show()

    return img * LoG

# Unpadding function
def unpad(img):
    M = len(img)
    N = len(img[0])
    return img[0:M//2, 0:N//2]


def freqLoGfilter(imagePath, sigma):
    L = 256

    # Read the image
    img = cv2.imread(imagePath)

    # Convert the image to HSV
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Grab the intensity portion of the image for processing
    intensityImg = img[:,:,2]

    # Pad the image
    paddedImg = pad(intensityImg)

    # Center the image
    centeredImg = center(paddedImg.astype(int))

    # FFT of the image
    f = np.fft.fft2(centeredImg)

    # Multiply with LoG filter
    ffiltered = LoG(f, sigma)

    magnitude_spectrum = 20*np.log(np.abs(ffiltered))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum of Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Inverse FFT
    img_back = np.fft.ifft2(ffiltered)

    # Uncenter the image
    img_back = center(img_back)

    # Take magnitude to eliminate complex numbers
    img_back = np.abs(img_back)

    # Unpad
    img_back = unpad(img_back)

    # Scale
    maxIntensity = np.amax(img_back)
    img_back = img_back * (L-1) / maxIntensity

    # Convert to unsigned int for display
    img_back = img_back.astype(np.uint8)

    # Recombine with Hue and Saturation values
    result = img.copy()
    result[:,:,2] = img_back

    # Convert back to BGR
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    cv2.imshow('Filtered Image', result)
    cv2.waitKey()

    return
