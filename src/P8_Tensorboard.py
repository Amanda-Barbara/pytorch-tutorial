from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import cv2

writer = SummaryWriter("logs")
image_path = "../imgs/dog.png"
img_PIL = Image.open(image_path)
img = cv2.imread(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("pil", img_array, 1, dataformats='HWC')
writer.add_image(tag="cv2", img_tensor=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), global_step=0, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()