from PIL import Image
import numpy as np
label = Image.open('/data/test/images/1120-2_30.bmp')
label = np.array(label)
print(label.shape)