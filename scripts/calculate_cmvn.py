# -*- coding: utf-8 -*-
# pylint: disable=import-error

"""calculate_cmvn.py: to compute cmvn separately."""

import sys
import numpy as np
from rnnt.base.util import Util

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

base_path = sys.argv[1]
feats = list()
with open(sys.argv[2]) as f:
  for path_file in f:
    feats.append(Util.get_fbanks(base_path + "/" +
                                 path_file.strip(),
                                 frame_size=0.025,
                                 frame_stride=0.01,
                                 n_filt=40))
feats = np.vstack(feats)
np.savetxt("fbank.mean", np.mean(feats, axis=0))
np.savetxt("fbank.std", np.std(feats, axis=0))
