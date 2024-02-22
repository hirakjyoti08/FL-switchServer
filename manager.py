import json
import re
from bs4 import BeautifulSoup
import requests

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

# Example usage:
manager = CPUScoreManager('properties.json')
manager.update_properties_with_cpu_scores()
manager.save_properties()