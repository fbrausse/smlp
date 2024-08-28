{
  "version": "1.2",
  "variables": [
      { "label": "o1", "interface": "output", "type": "real" },
      { "label": "o0",   "interface": "output", "type": "real" },
      { "label": "CH",     "interface": "input", "type": "int", "range": [0,1] },
      { "label": "RANK",   "interface": "input", "type": "int", "range": [0,1] },
      { "label": "Byte",   "interface": "input", "type": "int", "range": [0,7] },
      { "label": "p0", "interface": "knob", "type": "real", "range": [5,15], "grid": [5,6,7,8,9,10,11,12,13,14,15], "rad-rel": 0.1 },
      { "label": "p1", "interface": "knob", "type": "int", "range": [1,3], "grid": [1,2,3], "rad-rel": 0.1 },
      { "label": "p2", "interface": "knob", "type": "real", "range": [15,72], "grid": [15,20,24,30,35,40,45,50,55,60,65,72], "rad-rel": 0.1 },
      { "label": "p3", "interface": "knob", "type": "real", "range": [34,240], "grid": [34,40,38,60,80,120,240], "rad-rel": 0.1 }
  ],
  "objectives": {
      "objv1": "o0",
      "objv2": "o1"
  }
}
