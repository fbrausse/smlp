{
	"version": "1.2",
	"spec": [
		{
			"label": "rastrigin_max",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "x0",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-5.12,
				5.12
			]
		},
		{
			"label": "x1",
			"interface": "knob",
			"type": "real",
			"rad-rel": 0,
			"range": [
				-5.12,
				5.12
			]
		}
	],
	"objectives": {
		"objv": "rastrigin_max"
	}
}