import json

def update_json():
    with open('properties.json', 'r') as f:
        data = json.load(f)

    for key in data.keys():
        data[key]['done'] = False

    with open('properties.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    update_json()