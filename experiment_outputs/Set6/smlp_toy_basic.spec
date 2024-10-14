{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.001},
	{"label":"y", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.001},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-2 and x<=-0.5",
  "beta": "z<=6 and z>=5",
  "eta": "y<=1 and y>=-1",
  "objectives": {
    "objective": "z"
  }
}

