{
  "version": "1.2",
  "variables": [
      	{ "label": "Timing", "interface": "knob", "type": "int", "range": [-63,61], "rad-abs": 5 },
	{ "label": "CH",     "interface": "input", "type": "int", "range": [0,1] },
	{ "label": "RANK",   "interface": "input", "type": "int", "range": [0,1] },
	{ "label": "Byte",   "interface": "input", "type": "int", "range": [0,7] },
	{ "label": "i0",   "interface": "knob", "type": "real", "range": [40,120], "grid": [40,48,60,80,120], "rad-rel": 0.1 },
	{ "label": "i1",     "interface": "output", "type": "real" },
	{ "label": "i2",    "interface": "knob", "type": "real", "range": [15,72], "grid": [15,20,30,40,50,60,72], "rad-rel": 0.1 },
	{ "label": "i3",  "interface": "knob", "type": "real", "range": [7,15], "grid": [7,8,9,10,11,12,13,14,15], "rad-rel": 0.1 },
	{ "label": "i4", "interface": "knob", "type": "real", "range": [0,140], "grid": [0,35,70,140], "rad-rel": 0.1 },
	{ "label": "i5", "interface": "knob", "type": "real", "range": [0,60], "grid": [0,30,60], "rad-rel": 0.1 },
	{ "label": "o0", "interface": "output", "type": "real" }
  ],
  "alpha": "CH==0 and RANK==0 and Byte==0",
  "objectives": {
      "objv": "o0"
  }
}
