{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-2,2]},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-1 and x<=1",
  "beta": "z>=2.6 and z<=3.14",
  "eta": "y>=-1 and y<=1",
  "objectives": {
    "objective": "z"
  }
}

