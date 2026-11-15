import numpy as np
import ninewas9 as nw

print(f"NumPy version: {np.__version__}")
print(f"9was9 version: {nw.__version__}")

# Создаем фейковую картинку 100x200x3 (RGB)
img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

# 1. Resize
resized = nw.fast_resize(img, 50, 100)
print("resize:", resized.shape) # (50, 100, 3)

# 2. Downsample
down = nw.block_downsample(img, factor=2)
print("downsample:", down.shape) # (50, 100, 3)

# 3. Grayscale
gray = nw.rgb_to_grayscale(img)
print("grayscale:", gray.shape) # (100, 200)

# 4. BGR
bgr = nw.rgb_to_bgr(img)
print("bgr:", bgr.shape) # (100, 200, 3)
print("bgr check:", np.array_equal(img[:,:,0], bgr[:,:,2])) # True

# 5. Normalize
norm = nw.normalize_image(img)
print("normalize min/max:", norm.min() >= 0.0, norm.max() <= 1.0)

# 6. Pad
padded = nw.pad_image(img, 10, 10)
print("pad:", padded.shape) # (120, 220, 3)

print("All tests passed!")