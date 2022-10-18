SMLP = ../../../../smlp
SPEC = ../via_results.spec

.PHONY: train search

train:
	env -u DISPLAY $(SMLP)/src/train-nn.py -s $(SPEC) \
	  data -O $(OBJ) -e 500 |& tee train.log

search:
	/usr/bin/time -v stdbuf -oL $(SMLP)/src/prove-nn.py -b \
	  -D 0.05 -g model_gen_data_12.json -n 100 -O $(OBJ) \
	  -s $(SPEC) -S search.safe.csv -B data_bounds_data_12.json \
	  -t "[`LC_ALL=C seq 0.00 0.05 0.95 | tr '\n' ,`1]" -v \
	  model_complete_data_12.h5 2>&1 >$@.trace | tee $@.log

# use thresholds found individually, lowered by param L
search-all0:
	/usr/bin/time stdbuf -oL $(SMLP)/src/prove-nn.py -b \
	  -D 0.05 -g model_gen_data_12.json -n 100 -O $(OBJ) \
	  -s $(SPEC) -S search.safe.csv -B data_bounds_data_12.json \
	  -t "[`LC_ALL=C seq 0.00 0.05 0.95 | tr '\n' ,`1]" -v \
	  model_complete_data_12.h5 \
	  -a 'And(RL__8e_09Hz >= 
	  2>&1 >$@.trace | tee $@.log
