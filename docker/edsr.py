import os
import sys
import matplotlib.pyplot as plt

from data import DIV2K
from model.edsr import edsr

from utils import load_image
from model import resolve_single

lr_image_path = sys.argv[1]

# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

weights_dir = f'weights/edsr-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

model = edsr(scale=scale, num_res_blocks=depth)
model.load_weights(weights_file)

lr = load_image(lr_image_path)
sr = resolve_single(model, lr)

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(sr)
plt.savefig(sys.argv[1] + '_edsr.png', dpi=300, bbox_inches='tight', pad_inches=0)
