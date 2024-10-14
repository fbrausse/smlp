SMLP = ../../../../../smlp
SPEC = ../new.spec
LAYERS = 2,1

.PHONY: train search

train:
	env -u DISPLAY $(SMLP)/src/train-nn.py -s $(SPEC) \
	  data -O $(OBJ) -e 500 -l $(LAYERS) |& tee train.log

search:
	/usr/bin/time -v stdbuf -oL $(SMLP)/src/prove-nn.py -b \
	  -D 0.05 -g model_gen_data_12.json -n 100 -O $(OBJ) \
	  -s $(SPEC) -S search.safe.csv -B data_bounds_data_12.json \
	  -t "[`LC_ALL=C seq 0.00 0.05 0.95 | tr '\n' ,`1]" -v \
	  --alpha "$(ALPHA)" \
	  model_complete_data_12.h5 \
	  2>&1 >$@.trace | tee $@.log
