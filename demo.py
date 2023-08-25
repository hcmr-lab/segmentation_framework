import numpy as np
import matplotlib.pyplot as plt

# Create a grayscale image (2D array)
gray_image = np.random.rand(100, 100)

# Create an RGB image (3D array)
rgb_image = np.random.rand(100, 100, 3)

fig, axes = plt.subplots(1, 2)

# Display the grayscale image
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Grayscale Image')

# Display the color image
axes[1].imshow(rgb_image)
axes[1].set_title('RGB Image')

plt.show()