import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca

license_plate = np.invert(cca.plate_like_objs[0])
labelled_plate = measure.label(license_plate)
fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

chars = [] # store characters
count = 0
column_list = [] # track order of characters (starting x axis of each region)

for region in regionprops(labelled_plate):
    y0, x0, y1, x1 = region.bbox
    region_height = y1 - y0
    region_width = x1 - x0
    if (region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width): # valid letter (good size)
        ok_region = license_plate[y0:y1, x0:x1]

        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False) # draw rectangle

        ax1.add_patch(rect_border)

        resized_char = resize(ok_region, (20, 20)) # resize to 20x20
        chars.append(resized_char)

        column_list.append(x0)

plt.show()