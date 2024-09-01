{
	"version": "1.2",
	"spec": [
		{
			"label": "high_conditioned_elliptic_min",
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
				-2,
				2
			]
		}
	],
	"objectives": {
		"objv": "high_conditioned_elliptic_min"
	}
}
