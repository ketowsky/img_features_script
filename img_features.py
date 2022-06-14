import cv2
import urllib.request
import numpy as np
from matplotlib import pyplot as plt
import skimage

# Get IMG
url = "https://storage.googleapis.com/symfoni-prod-gwphosting-no/2020/01/tcp-girls_chatting.jpg"
url_response = urllib.request.urlopen(url)
img = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)

# Get basic image parameters
img_size_height, img_size_width, img_size_channels = img.shape
img_brightness = img.mean()


# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
graycom = skimage.feature.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

# Find the GLCM properties
GLCM_contrast = skimage.feature.greycoprops(graycom, 'contrast')
GLCM_dissimilarity = skimage.feature.greycoprops(graycom, 'dissimilarity')
GLCM_homogeneity = skimage.feature.greycoprops(graycom, 'homogeneity')
GLCM_correlation = skimage.feature.greycoprops(graycom, 'correlation')
GLCM_ASM = skimage.feature.greycoprops(graycom, 'ASM')


# RESULTS
result_dict = {
    'img_size': {
        'height': img_size_height,
        'width': img_size_width,
        'channels': img_size_channels
    },
    'GLCM': {
        'contrast': GLCM_contrast,
        'dissimilarity': GLCM_dissimilarity,
        'homogeneity': GLCM_homogeneity,
        'correlation': GLCM_correlation,
        'ASM': GLCM_ASM
    },
    'img_brightness': img_brightness
}

print(result_dict)
for element in result_dict:
    print(element + ": " + str(result_dict[element]))

# CALCULATE AND PLOT HISTOGRAM
color = ('b','g','r')
plt_list = []
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.subplot(221+i)
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.show()
