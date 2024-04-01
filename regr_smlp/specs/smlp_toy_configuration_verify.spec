{
    "version": "1.1",
    "spec":[
	{"label": "y1", "type": "response", "range": "float"},
	{"label": "y2", "type": "response", "range": "float"},
	{"label": "x", "type": "input", "range": "float", "bounds": [0,10]},
	{"label": "p1", "type": "knob", "range": "float", "rad-rel": 0.1, "grid": [2,4,7], "bounds": [0,10]},
	{"label": "p2", "type": "knob", "range": "float", "rad-abs": 0.2, "bounds": [3,7]}
    ],
    "configurations": {
	"stable_config": {
	    "p1": 7.0,
	    "p2": 6.000000067055225
	},
        "grid_conflict": {
            "p1": 3.0,
            "p2": 6.000000067055225
	},
        "unstable_config": {
            "p1": 7.0,
            "p2": 6.0
	},
	"not_feasible": {
            "p1": 7.0,
            "p2": 6.0
	}
    }
}
