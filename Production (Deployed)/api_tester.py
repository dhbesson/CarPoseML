import requests
import base64

url = "https://uru0x7k769.execute-api.us-west-1.amazonaws.com/test/carlambda"

upload_filename = './ID_00ab59fa6.jpg'
payload={}
files=[
  ('pic',(upload_filename,open(upload_filename,'rb'),'image/jpeg'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)
body_dec = base64.b64decode(response.text)

response_filename = 'response_img.jpg'  
with open(response_filename, 'wb') as f:
    f.write(body_dec)
