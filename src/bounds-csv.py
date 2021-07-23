#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(sys.stdin)
sc = MinMaxScaler()
sc.fit(data)
pd.DataFrame(((t.type(v) for t,v in zip(data.dtypes, l))
              for l in (sc.data_min_,sc.data_max_)),
             columns=data.columns,
             index=('min','max')).to_csv(sys.stdout, index=True)
