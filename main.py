import requests
from urllib.parse import quote

print(quote('Harry Potter'))

# https://en.wikipedia.org/w/api.php?action=query&page=Harry_Potter
resp = requests.get("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&titles="

                    + quote('Harry Potter'))
page = list(resp.json()['query']['pages'].values())[0]
print(page['extract'])

