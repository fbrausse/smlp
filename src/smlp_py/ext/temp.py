import json

# Path to your JSON file
json_file_path = 'smlp_toy_basic.spec'

# Open and read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)  # Load JSON data into a dictionary

# Print all dictionary items (key-value pairs)
for key, value in data.items():
    if key == 'variables':
        #print(f"{key}: {value}")
        for v in value:
            #print(v)
            for ite, val in v.items():
                if str(ite) == "rad-abs":
                    print(f"For {v["label"]} {ite}: {val}")

