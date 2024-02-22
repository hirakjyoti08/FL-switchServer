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
            for i, client in enumerate(data, start=1):
                # Generate a unique node_id
                node_id = str(uuid.uuid4())
                data[client]['node_id'] = node_id
                data[client]['node_number'] = f"node{i}"  # Add the node number
                if 'done' not in data[client]:
                    data[client]['done'] = False  # Add the 'done' attribute
            f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)
            f.truncate()     # remove remaining part
            print(f"Node IDs, node numbers, and 'done' attributes have been added to properties.json")

    def get_server_node(self):
        with open(self.properties_file, 'r+') as f:
            data = json.load(f)
            max_score = 0
            server_node = None
            server_key = None
            for client in data:
                if data[client]['done'] == False:  # Ignore the nodes that are marked as 'done'
                    score = int(data[client]['properties']['CPU Mark Score'].replace(',', ''))
                    if score > max_score:
                        max_score = score
                        server_node = data[client]['node_number']
                        server_key = client  # Keep track of the actual key
            if server_key:
                data[server_key]['done'] = True  # Mark the selected node as 'done'
                f.seek(0)        # <--- should reset file position to the beginning.
                json.dump(data, f, indent=4)
                f.truncate()     # remove remaining part
            return server_node
            
manager = CPUScoreManager('properties.json')
manager.update_properties_with_cpu_scores()
manager.save_properties()
manager.generate_uuid()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', help='Execute server.py in a random node')
    parser.add_argument('--compute', action='store_true', help='Execute server.py from properties.json')
    args = parser.parse_args()

    node = None

    if args.random:
        node = random.choice(['node1', 'node2'])
        print(f"Chosen node: {node}")
        subprocess.run(["python", os.path.join(node, "server.py")])
    elif args.compute:
        manager.generate_uuid()
        server_node = manager.get_server_node()
        print(f"Server node: {server_node}")
        subprocess.run(["python", os.path.join(server_node, "server.py")])

    # Start the client.py files for the other nodes, Get a list of all nodes
    nodes = [dir for dir in next(os.walk('./'))[1] if dir.startswith('node')]
    # Start the client.py files for the other nodes
    for other_node in nodes:
        if other_node != node:
            print(f"Starting client: {other_node}")
            subprocess.run(["python", os.path.join(other_node, "client.py")])

if __name__ == "__main__":
    main()
