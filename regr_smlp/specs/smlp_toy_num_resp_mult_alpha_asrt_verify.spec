{"version": "1.1", "spec":[{"label": "y1", "type": "response", "range": "float"}, {"label": "y2", "type": "response", "range": "float"}, {"label": "x", "type": "input", "range": "float", "bounds": [0,10]}, {"label": "p1", "type": "input", "range": "float", "bounds": [0,10]}, {"label": "p2", "type": "input", "range": "float", "bounds": [3,7]}], "alpha":"p2==7.0 and x==0 and p1==2.5", "assertions":{"asrt1":"(y2**3+p2)/2<6", "asrt2":"y1>=9", "asrt3":"y2<0"}}
