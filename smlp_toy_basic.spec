{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-1,5], "rad-abs":0.001},
	{"label":"y", "interface":"knob", "type":"real", "range":[-0.5,2], "rad-abs":0.001},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "beta": "z<=6 and z>=4",
  "eta": "y<=0.5 and y>=-0.5",
  "objectives": {
    "objective": "z"
  }
}

