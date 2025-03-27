import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open("3DBenchy.png"))

gray = np.mean(img, axis=2).astype(np.uint8)

plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.show()

Image.fromarray(gray).save("Gray3DBenchy.png")
