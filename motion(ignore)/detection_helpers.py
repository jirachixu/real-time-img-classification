import numpy as np
from union_find import union_find

# implemented using union find for very efficient labeling
# returns list of bounding boxes [xmin, ymin, xmax, ymax]
def connected_components(mask):
    height, width = mask.shape
    uf = union_find()
    
    def index(x, y):
        return y * width + x
    
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            uf.make_set(index(x, y))
            if x < width - 1 and mask[y, x + 1]:
                uf.make_set(index(x + 1, y))
                uf.union_sets(index(x, y), index(x + 1, y))
            if y < height - 1 and mask[y + 1, x]:
                uf.make_set(index(x, y + 1))
                uf.union_sets(index(x, y), index(x, y + 1))
    
    components = {}
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            root = uf.find_set(index(x, y))
            if root not in components:
                components[root] = [x, y, x, y]
            else:
                xmin = min(components[root][0], x)
                ymin = min(components[root][1], y)
                xmax = max(components[root][2], x)
                ymax = max(components[root][3], y)
                components[root] = [xmin, ymin, xmax, ymax]
    
    return list(components.values())

# unused functions for convolution and gaussian blur, opencv functions are used instead because 
# they are optimized, tried using these but too slow for real-time processing
def gaussian_blur(kernel_size=5, sigma=1.0):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    
    # normalize to maintain brightness of image
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    height, width = image.shape
    kheight, kwidth = kernel.shape
    # pad to maintain image size, otherwise edges get smaller (edges of kernel go "outside" the image)
    pad_h = kheight // 2
    pad_w = kwidth // 2
    
    # adds pad_h rows of pixels on top and bottom, pad_w columns on left and right
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + kheight, j:j + kwidth]
            output[i, j] = np.sum(region * kernel)
            
    return output