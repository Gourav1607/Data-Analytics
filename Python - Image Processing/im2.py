import cv2

image = cv2.imread('img2.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(image, (7,7), 0)

cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
 
cv2.waitKey(0)
