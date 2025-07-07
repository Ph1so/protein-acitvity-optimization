import requests
import time
import os
import json

public_dir = r"C:\Users\Public"
subdir = os.path.join(public_dir, "Public BayesianOpt")
os.makedirs(subdir, exist_ok=True)

# Data to write
data = {
    "pH": 1,
    "enzyme_conc": 2,
    "incubation_time": 3
}

# File path
file_path = os.path.join(subdir, "params.json")

# Write JSON to file
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)  # `indent=4` makes it nicely formatted
# data = {
#   "username": "admin",
#   "password": "genera"
# }

# headers = {
#     "accept": "*/*",
#     "Content-Type": "application/json"
# }

# response = requests.post("http://192.168.0.101:8086/genera/login", json = data, headers=headers)
# session_id = response.headers["Set-Cookie"]
# session_id = session_id[session_id.index("=")+ 1: session_id.index(";")]
# print(f"header:\n{response.headers}\ncontent:\n{response.cookies}")

# process = {
#   "processId": "C:\\retisoft\\genera\\processes\\move_plate2.process",
#   "name": "move_plate2",
#   "batchCount": 1,
#   "priority": 1
# }
# headers = {
#     "accept": "application/json",
#     "cookie": f"session={session_id}"
# }
# response = requests.post("http://192.168.0.101:8086/genera/scheduler/process-tasks", json = process, headers=headers)

# try:
#     response_data = response.json()
#     print("Process Submission Response:", response_data)
    
#     # Safely extract ID
#     process_id = response_data.get("id")
#     if process_id is None:
#         raise ValueError("Process ID not found in response.")
# except ValueError as ve:
#     print("Error:", ve)
#     exit(1)
# except Exception as e:
#     print("Unexpected error:", e)
#     exit(1)

# headers = {
#     "cookie": f"session={session_id}"
# }
# print(process_id)
# state = "RUNNING"

# while state == "RUNNING":
#     response = requests.get(f"http://192.168.0.101:8086/genera/scheduler/process-tasks/{process_id}/state", headers=headers)
#     state = response.text[1:-1]
#     print(state)
#     time.sleep(2)