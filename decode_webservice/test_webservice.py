import requests
from PIL import Image
import io

# Endpoint of your Sanic server
URL = "http://localhost:8050/mask"

# Your payload (make sure this is valid RLE data)
payload = {
    "size": [600, 800],
    "counts": "`l]74db0:F5K7I5K:F6J0000000000000000000000O10000000000O100000000O100000000000000O1000000000000O10000000000O100000000O10000000000O100000000000000O1000000000000O10000000000O10000000000000000O10000O100O10000O10000000000000002N2N003M3L9H4K4MgkR5"
}

# Send POST request
response = requests.post(URL, json=payload)

# Check response
if response.status_code == 200:
    # Load image from binary content
    image = Image.open(io.BytesIO(response.content))
    image.show()  # Opens default viewer
else:
    print(f"Failed to get image: {response.status_code}")
    print(response.text)
