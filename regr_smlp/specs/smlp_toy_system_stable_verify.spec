{
  "version": "1.2",
  "variables": [
      {"label":"y1", "interface":"output", "type":"real"},
      {"label":"y2", "interface":"output", "type":"real"},
      {"label":"x1", "interface":"input", "type":"real", "range":[0,10]},
      {"label":"x2", "interface":"input", "type":"real", "range":[0,10]},
      {"label":"p1", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.2, "grid":[0]},
      {"label":"p2", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0, "grid":[0]}
  ],
  "system": {
      "y1":"0 if p1<=0 and x1+x2+p2>-1 else p1",
      "y2":"0 if p2<=0 and x1+x2+p1>-1 else p2"
  },
  "assertions": {
    "assert1": "y1<=0",
    "assert2": "y2<=0"
  }
}
