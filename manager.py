import json
import re
from bs4 import BeautifulSoup
import requests

import argparse
import random
import subprocess
import os
import uuid

class CPUScoreManager:
    def __init__(self, properties_file):
        self.properties_file = properties_file
        self.properties = self.load_properties()

    def load_properties(self):
        with open(self.properties_file, 'r') as f:
            return json.load(f)

    def clean_cpu_name(self, cpu_name):
        cpu_name = re.sub(r'\(R\)|\(TM\)', '', cpu_name)
        cpu_name = re.sub(r' CPU', '', cpu_name)
        cpu_name = re.sub(r'\s+', ' ', cpu_name).strip()
        return cpu_name

    def get_cpu_mark_score(self, cpu_name):
        url = 'https://www.cpubenchmark.net/cpu_list.php'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'cputable'})
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if cols[0].text.strip() == cpu_name:
                return cols[1].text.strip()
        return None

    def update_properties_with_cpu_scores(self):
        for key in self.properties:
            cpu_name = self.properties[key]['properties']['Full CPU name']
            cleaned_cpu_name = self.clean_cpu_name(cpu_name)
            cpu_mark_score = self.get_cpu_mark_score(cleaned_cpu_name)
            if cpu_mark_score is not None:
                self.properties[key]['properties']['CPU Mark Score'] = cpu_mark_score

    def save_properties(self):
        with open(self.properties_file, 'w') as f:
            json.dump(self.properties, f, indent=4)

    def generate_uuid(self):
        with open(self.properties_file, 'r+') as f:
            data = json.load(f)
            for client in data:
                # Generate a unique node_id
                node_id = str(uuid.uuid4())
                data[client]['node_id'] = node_id
            f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)
            f.truncate()     # remove remaining part
            
manager = CPUScoreManager('properties.json')
manager.update_properties_with_cpu_scores()
manager.save_properties()
manager.generate_uuid()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', help='Execute server.py in a random node')
    parser.add_argument('--compute', action='store_true', help='Execute server.py from properties.json')

    args = parser.parse_args()

    if args.random:
        node = random.choice(['node1', 'node2'])
        print(f"Chosen node: {node}")
        subprocess.run(["python", os.path.join(node, "server.py")])
    elif args.compute:
        with open('properties.json') as f:
            data = json.load(f)
            client_node = data.get('client')
            if client_node:
                print(f"Client node: {client_node}")
                subprocess.run(["python", os.path.join(client_node, "server.py")])
            else:
                print("No client specified in properties.json")

if __name__ == "__main__":
    main()
