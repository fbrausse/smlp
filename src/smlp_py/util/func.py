# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

def const(v):
	return lambda *args, **kwargs: v
