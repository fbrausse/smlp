{
	"version": "1.2",
	"variables": [
		{
			"label": "y1",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "y2",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "x",
			"interface": "input",
			"type": "real",
			"range": [
				0,
				10
			]
		},
		{
			"label": "p1",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0.1,
			"grid": [
				2,
				4,
				7
			],
			"range": [
				0,
				10
			]
		},
		{
			"label": "p2",
			"interface": "knob",
			"type": "real",
			"rad-abs": 0.2,
			"range": [
				3,
				7
			]
		}
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