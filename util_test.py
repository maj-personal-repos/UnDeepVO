import pykitti
from math import pi
from util import euler_to_rotation, position_to_translation
import numpy as np

dataset = pykitti.odometry('data/dataset', '01')


rotm = euler_to_rotation([0, 0, pi / 4])

transm = position_to_translation([1, 1, 0])

K = dataset.calib.K_cam2

K