{
  "version": "1.2",
  "variables": [
      {"label":"y", "interface":"output", "type":"real"},
      {"label":"y_copy", "interface":"output", "type":"real"},
      {"label":"x", "interface":"input", "type":"real", "range":[0,10]},
      {"label":"x_copy", "interface":"input", "type":"real", "range":[0,10]},
      {"label":"p", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0, "grid": [0.05]}
  ],
  "system": {
      "y":"0 if p<=0 else x",
      "y_copy":"0 if p<=0 else x_copy"
  },
  "assertions": {
    "assert_incr": "x > x_copy or y <=y_copy",
    "assert_decr": "x > x_copy or y >=y_copy"
  }
}
