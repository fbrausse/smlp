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