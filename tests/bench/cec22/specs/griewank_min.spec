{
	"version": "1.2",
	"spec": [
		{
			"label": "griewank_min",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "x0",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-20,
				20
			]
		},
		{
			"label": "x1",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-20,
				20
			]
		}
	],
	"objectives": {
		"objv": "griewank_min"
	}
}