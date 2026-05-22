{
	"version": "1.2",
	"spec": [
		{
			"label": "ackley_max",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "x0",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-5,
				5
			]
		},
		{
			"label": "x1",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-5,
				5
			]
		}
	],
	"objectives": {
		"objv": "ackley_max"
	}
}