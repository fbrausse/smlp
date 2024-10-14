{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-1.5 and x<=1.5",
  "beta": "z>=3 and z<=4",
  "eta": "y>=-2 and y<=2",
  "objectives": {
    "objective": "z"
  }
}

