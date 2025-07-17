from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import matplotlib.patches as patches
from matplotlib.lines import Line2D

results = []
downward = glob.glob("./frames/*.png")
downward.sort()
downward = [cv2.imread(s, cv2.IMREAD_GRAYSCALE) for s in downward]
images = downward

n_images = len(images)
n_cols = min(5, n_images)  # Max 5 columns
n_rows = (n_images + n_cols - 1) // n_cols 

plt.figure(figsize=(6*n_cols, 4*n_rows))

for i, img in enumerate(images):
    ax1 = plt.subplot(n_rows, n_cols*2, i*2+1)
    ax2 = plt.subplot(n_rows, n_cols*2, i*2+2)
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Original Image {i+1}')
    ax1.axis('off')
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_valid = False
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        min_area = 100  
        min_dimension = 30  
        
        is_long_enough = (w > min_dimension or h > min_dimension)
        
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            is_line_like = circularity < 0.5
            
            if area > min_area and is_long_enough and is_line_like:
                line_valid = True
    ax2.imshow(thresh, cmap='gray')
    ax2.set_title('with contours')
    ax2.axis('off')

    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(thresh_rgb, contours, -1, (0, 255, 0), 2)
    ax2.imshow(thresh_rgb)
    if line_valid:
        points = np.argwhere(thresh > 0)
        if len(points) > 10:
            points = points.astype(np.float64)
            xs = points[:, 1]
            ys = points[:, 0]
            
            # Determine if the line is more vertical or horizontal
            x_range = np.max(xs) - np.min(xs)
            y_range = np.max(ys) - np.min(ys)
            is_vertical = y_range > x_range
            
            if is_vertical:
                ransac_points = np.column_stack((ys, xs))
                
                from sklearn.linear_model import RANSACRegressor
                ransac = RANSACRegressor(min_samples=2, residual_threshold=5.0, random_state=42)
                ransac.fit(ransac_points[:, 0].reshape(-1, 1), ransac_points[:, 1])
                
                slope = ransac.estimator_.coef_[0]
                intercept = ransac.estimator_.intercept_
                
                h, w = img.shape
                y1, x1 = 0, intercept
                y2, x2 = h-1, slope * (h - 1) + intercept
                x1 = max(0, min(w-1, x1))
                x2 = max(0, min(w-1, x2))
            else:
                ransac_points = np.column_stack((xs, ys))
                
                from sklearn.linear_model import RANSACRegressor
                ransac = RANSACRegressor(min_samples=2, residual_threshold=5.0, random_state=42)
                ransac.fit(ransac_points[:, 0].reshape(-1, 1), ransac_points[:, 1])
                
                slope = ransac.estimator_.coef_[0]
                intercept = ransac.estimator_.intercept_
                
                h, w = img.shape
                x1, y1 = 0, intercept
                x2, y2 = w-1, slope * (w - 1) + intercept
                y1 = max(0, min(h-1, y1))
                y2 = max(0, min(h-1, y2))

            line = Line2D([x1, x2], [y1, y2], color='red', linewidth=2)
            ax1.add_line(line)
            ax1.set_title(f'line in image {i+1}')
        else:
            ax1.set_title(f'no valid line in image {i+1}')
    else:
        ax1.set_title(f'no valid rope {i+1}')

    results.append(line_valid)

plt.tight_layout()
plt.show()
print(results)



