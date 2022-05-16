import os
import sys
import matplotlib.pyplot as plt

from data import DIV2K
from model.srgan import generator

from utils import load_image
from model import resolve_single

lr_image_path = sys.argv[1]

weights_file = 'weights/srgan/gan_generator.h5'

model = generator()
model.load_weights(weights_file)

lr = load_image(lr_image_path)
sr = resolve_single(model, lr)

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(sr)
plt.savefig(sys.argv[1] + '_srgan.png', dpi=300, bbox_inches='tight', pad_inches=0)
