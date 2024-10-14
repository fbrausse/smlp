{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.01},
	{"label":"y", "interface":"knob", "type":"real", "range":[-2,2]},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-1.5 and x<=0",
  "beta": "z<=2.1 and z>=1",
  "eta": "y>=-2 and y<=0",
  "objectives": {
    "objective": "z"
  }
}

