{
    "version": "1.1",
    "spec":[
	{"label": "y1", "type": "response", "range": "float"},
	{"label": "y2", "type": "response", "range": "float"},
	{"label": "x", "type": "input", "range": "float", "bounds": [0,10]},
	{"label": "p1", "type": "knob", "range": "float", "rad-rel": 0.1, "grid": [2,4,7], "bounds": [0,10]},
	{"label": "p2", "type": "knob", "range": "float", "rad-abs": 0.2, "bounds": [3,7]}
    ],
    "witnesses": {
	"stable_witness": {
	    "x": 7,
	    "p1": 7.0,
	    "p2": 6.000000067055225
	},
        "grid_conflict": {
	    "x": 6.2,
            "p1": 3.0,
            "p2": 6.000000067055225
	},
        "unstable_witness": {
	    "x": 7,
            "p1": 7.0,
            "p2": 6.0
	},
	"infeasible_witness": {
	    "x": 7,
            "p1": 7.0,
            "p2": 6.0
	}
    }
}
