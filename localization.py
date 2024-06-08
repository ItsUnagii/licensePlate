from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# License plate detection
car_img = imread("car2.jpg", as_gray=True)
print(car_img.shape)

graycar_img = car_img * 255 # imread ranges between 1 and 0 I think
fig, (ax1, ax2) = plt.subplots(1,2) # make 2 plots!
ax1.imshow(graycar_img, cmap="gray") # show image 1 on one side
threshold_val = threshold_otsu(graycar_img) # threshold
binarycar_img = graycar_img > threshold_val # if bigger than threshold be white
ax2.imshow(binarycar_img, cmap="gray")
plt.show()


