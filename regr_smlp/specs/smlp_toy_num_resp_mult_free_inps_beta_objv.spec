{
    "version": "1.1",
    "spec": [
	{"label": "y1", "type": "response", "range": "float"},
	{"label": "y2", "type": "response", "range": "float"},
	{"label": "x", "type": "input", "range": "float", "bounds":[-2, null]},
	{"label": "p1", "type": "knob", "range": "float", "rad-rel": 0.1, "grid": [2,4,7], "bounds":[4,8]},
	{"label": "p2", "type": "knob", "range": "float", "rad-abs": 0.2, "bounds": [3,7]}],
    "beta": "y1>7 and y2>6",
    "objectives": {"obj1": "(y1+y2)/2","objv2": "y1/2-y2","objv3": "y2"}
}
