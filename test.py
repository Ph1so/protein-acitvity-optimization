import requests

data = {
  "username": "admin",
  "password": "genera"
}

headers = {
    "accept": "*/*",
    "Content-Type": "application/json"
}

response = requests.post("http://192.168.0.101:8086/genera/login", json = data, headers=headers)
session_id = response.headers["Set-Cookie"]
session_id = session_id[session_id.index("=")+ 1: session_id.index(";")]
print(f"header:\n{response.headers}\ncontent:\n{response.cookies}")

process = {
  "processId": "C:\\retisoft\\genera\\processes\\move_plate2.process",
  "name": "move_plate2",
  "batchCount": 1,
  "priority": 1
}
headers = {
    "accept": "application/json",
    "cookie": f"session={session_id}"
}
response = requests.post("http://192.168.0.101:8086/genera/scheduler/process-tasks", json = process, headers=headers)
print(response.json())