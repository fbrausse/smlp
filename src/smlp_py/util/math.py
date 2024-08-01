# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import numpy as np

def cartesian(*xs):
	return np.stack(np.meshgrid(*map(np.asarray, xs)), -1).reshape(-1, len(xs))

__all__ = [s.__name__ for s in (cartesian,)]
