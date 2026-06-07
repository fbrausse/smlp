{
	"version": "1.2",
	"variables": [
		{
			"label": "o0",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "o1",
			"interface": "output",
			"type": "real"
		},
		{
			"label": "CH",
			"interface": "input",
			"type": "int",
			"range": [
				0,
				1
			]
		},
		{
			"label": "RANK",
			"interface": "input",
			"type": "int",
			"range": [
				0,
				1
			]
		},
		{
			"label": "Byte",
			"interface": "input",
			"type": "int",
			"range": [
				0,
				7
			]
		},
		{
			"label": "p0",
			"interface": "knob",
			"type": "real",
			"range": [
				40,
				240
			],
			"grid": [
				40,
				48,
				60,
				80,
				120,
				240
			],
			"rad-rel": 0.1
		},
		{
			"label": "p1",
			"interface": "knob",
			"type": "real",
			"range": [
			    21,
			    83],
			"grid": [
			    21,
			    25,
			    28,
			    34,
			    40,
			    45,
			    52,
			    56,
			    63,
			    74,
			    83
			    ]
		},
		{
			"label": "p2",
			"interface": "knob",
			"type": "real",
			"range": [
				40,
				80
			],
			"grid": [
				40,
				48,
				60,
				80
			],
			"rad-rel": 0.1
		},
		{
			"label": "p3",
			"interface": "knob",
			"type": "real",
			"range": [
				7,
				15
			],
			"grid": [
				7,
				8,
				9,
				10,
				11,
				12,
				13,
				14,
				15
			],
			"rad-rel": 0.1
		},
		{
			"label": "p4",
			"interface": "knob",
			"type": "real",
			"range": [
				0,
				420
			],
			"grid": [
				0,
				35,
				70,
				140,
				210,
				280,
				350,
				420
			],
			"rad-rel": 0.1
		},
		{
			"label": "p5",
			"interface": "knob",
			"type": "real",
			"range": [
				0,
				2400
			],
			"grid": [
				0,
				30,
				60,
				500,
				600,
				900,
				1400,
				2400
			],
			"rad-rel": 0.1
		}
	],
	"objectives": {
	    "objv0": "o0",
	    "objv1": "o1"
	}
}
