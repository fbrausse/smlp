{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[0,6]},
	{"label":"y", "interface":"knob", "type":"real", "range":[-3,3]},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>2 and x<4",
  "beta": "z<2.1",
  "eta": "y<1 and y>-1",
  "objectives": {
    "objective": "z"
  }
}

