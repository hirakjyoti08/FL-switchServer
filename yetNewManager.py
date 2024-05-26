import time
import requests
import threading
import json
import subprocess
import socket

with open('properties.json') as f:
    properties = json.load(f)


def wait_for_properties_update():
    while True:
        try:
            curr_server = get_lowest_ranked_server_ip(properties)
            response = requests.get(
                f'http://{curr_server}:5000/properties-updated', timeout=5)
            if response.status_code == 200:
                properties[curr_server]['isServer'] = True
                break  # The properties.json file has been updated, so we can stop waiting
        except requests.exceptions.RequestException:
            pass  # Ignore exceptions and keep waiting


def check_server_heartbeat(server_ip: str):
    while True:
        try:
            response = requests.get(f'http://{server_ip}:5000/heartbeat', timeout=5)
            if response.status_code != 200:
                print(f'Server {server_ip} is dead')
                # Handle server death...
                notify_and_handle_server_death()
                subprocess.run(['python', 'server.py'])
        except requests.exceptions.RequestException:
            print(f'Server {server_ip} is dead')
            # Handle server death...
            notify_and_handle_server_death()
        time.sleep(5)

def notify_and_handle_server_death():
    properties[get_host_ip()]['isServer'] = True
    # send signal that properties.json has been updated
    for ip in properties.keys():
        if ip == get_host_ip():
            continue
        try:
            post_response = requests.post(f'http://{ip}:5000/properties-updated', timeout=5)
            if post_response.status_code != 200:
                print(f'Failed to send update signal to server {ip}')
        except requests.exceptions.RequestException:
            print(f'Failed to send update signal to server {ip}')

def get_lowest_ranked_server_ip(properties):
    servers = {ip: info for ip, info in properties.items() if info['isServer']}
    lowest_ranked_server_ip = max(servers, key=lambda ip: servers[ip]['rank'])
    return lowest_ranked_server_ip

# run as client
def run_client():
    wait_for_properties_update()
    with open('properties.json') as f:
        properties = json.load(f)
    lowest_ranked_server_ip = get_lowest_ranked_server_ip(properties)
    subprocess.run(['python', 'client.py', '--ip=' + lowest_ranked_server_ip])

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        print("Unable to get Hostname and IP")
        return None

def my_server_rank() -> int:
    my_ip = get_host_ip()
    return properties[my_ip]["rank"] if my_ip in properties else None

def is_next_server():
    previous_node = [node for node in properties.values() if node["rank"] == my_server_rank - 1]
    if not previous_node:
        return False
    return previous_node[0]["isServer"], previous_node[0]["IP address"]



isNextServer, prev_server_ip = is_next_server()

if __name__ == "__main__":
    if isNextServer:
        check_server_heartbeat(prev_server_ip)
    else:
        run_client()
            













# check for heartbeat
# heartbeat_check_thread = threading.Thread(target=check_server_heartbeat, args=("192.168.137.107",))
# heartbeat_check_thread.start()