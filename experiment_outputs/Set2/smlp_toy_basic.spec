{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[1.5,4.5], "rad-abs":0.01},
	{"label":"y", "interface":"knob", "type":"real", "range":[2.5,4.5], "rad-abs":0.01},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=2.5 and x<=3.5",
  "beta": "z>=1.5",
  "eta": "y<3.5",
  "objectives": {
    "objective": "z"
  }
}

