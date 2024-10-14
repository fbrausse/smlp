{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[0,3.5]},
	{"label":"y", "interface":"knob", "type":"real", "range":[0,6]},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=1 and x<=3",
  "beta": "z>=2 and z<=4.4",
  "eta": "y>=2 and y<=4.5",
  "objectives": {
    "objective": "z"
  }
}

