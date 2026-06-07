{
  "version": "1.2",
  "variables": [
    {"label":"y", "interface":"output", "type":"real"},
    {"label":"x", "interface":"input", "type":"real", "range":[-1,1]},
    {"label":"p", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.2}
  ],
  "system": {
    "y":"0 if (p<=0 and x<=0) else x if x>0 else p"
  },
  "queries": {
      "query_stable_witness": "y<=0",
      "query_unstable_witness": "y<=0",
      "query_non_witness_neg": "y<=0",
      "query_non_witness_pos": "y<=0"
  },
  "witnesses": {
    "query_stable_witness": {"x": -0.5, "p":-1},
    "query_unstable_witness": {"x": -0.5, "p":0},
    "query_non_witness_neg": {"x": -0.5, "p":1},
    "query_non_witness_pos": {"x":  0.5, "p":1}
  }
}
