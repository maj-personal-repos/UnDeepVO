import pykitti
from math import pi
from util import eul2rotm, pos2transm
import numpy as np

dataset = pykitti.odometry('data/dataset', '01')


rotm = eul2rotm([0, 0, pi/4])

transm = pos2transm([1, 1, 0])

K = dataset.calib.K_cam2

K