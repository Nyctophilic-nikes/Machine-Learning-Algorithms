import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:\\Users\\HP\\Downloads\\peppers.png")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
segmented = cv2.pyrMeanShiftFiltering(img, 21, 51)
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.show()
