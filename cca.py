# cca - connected component analysis
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization # other file

# groups connected regions?
label_image = measure.label(localization.binarycar_img)

# make license plate parameter shapes
plateDimensions = (0.04*label_image.shape[0], 0.2*label_image.shape[0], 0.08*label_image.shape[1], 0.2*label_image.shape[1])
minHeight, maxHeight, minWidth, maxWidth = plateDimensions
plate_obj_coords = []
plate_like_objs = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(localization.graycar_img, cmap="gray")

# regionprops build properies of measure's labelled regions
for region in regionprops(label_image):
    if (region.area < 50):
       # small areas are probably not license plates
        continue 
        
    minRow, minCol, maxRow, maxCol = region.bbox
    regionHeight = maxRow - minRow
    regionWidth = maxCol - minCol
    if (regionHeight >= minHeight and regionHeight <= maxHeight and regionWidth >= minWidth and regionWidth <= maxWidth and regionWidth > regionHeight):
        plate_like_objs.append(localization.binarycar_img[minRow:maxRow, minCol:maxCol])
        plate_obj_coords.append((minRow, minCol, maxRow, maxCol))
        rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=1, fill=False)
        ax1.add_patch(rectBorder)
    

plt.show()