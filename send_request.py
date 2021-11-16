import json
import requests

input_json = json.dumps({'age': '42','price':'50000'})
res = requests.post('http://127.0.0.1:8000/predict',input_json)

print(res.text)