{
	"version": "1.2",
	"spec": [
		{
			"label": "rosenbrock_max",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "x0",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-2,
				2
			]
		},
		{
			"label": "x1",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-1,
				3
			]
		}
	],
	"objectives": {
		"objv": "rosenbrock_max"
	}
}