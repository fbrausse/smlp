{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-1.5,1.5], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-1.5,1.5], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=0.5 and x<=1.5",
  "beta": "z>=0.5 and z<=1",
  "eta": "y>=0 and y<=-1.5",
  "objectives": {
    "objective": "z"
  }
}

