{
  "version": "1.2",
  "variables": [
      {"label":"y1", "interface":"output", "type":"real"},
      {"label":"y2", "interface":"output", "type":"real"},
      {"label":"p1", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.2, "grid":[0]},
      {"label":"p2", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0, "grid":[0]}
  ],
  "beta": "y2<=0",
  "system": {
      "y1":"0 if p1<=0 and p2>-1 else p1",
      "y2":"0 if p2<=0 and p1>-1 else p2"
  },
  "assertions": {
      "assert": "y2<=0"
  },
  "objectives": {
      "objv1": "y1",
      "objv2": "y1*y2"
  }
}
