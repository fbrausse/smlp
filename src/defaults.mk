#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# general
RESP = delta
OBJT = $(RESP)

# train
TRAIN_BATCH  = 32
TRAIN_EPOCHS = 30
TRAIN_LAYERS = 2,1
TRAIN_SEED   = 1234
TRAIN_FILTER = 0

# verify
T  = 90
ST = 95
N  = 1

# search
COFF = 0.05
TLO = 0
THI = 0.9
$(SEARCH).safe.csv: T = '[$(shell seq -s , $(TLO) $(COFF) $(THI))]'
