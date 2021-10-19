import requests
print(requests.get('http://127.0.0.1:5000/',{
    'chat':'I hate this product'
}).json()
)
print(requests.get('http://127.0.0.1:5000/',{
    'chat':'I really love this product'
}).json()
)

print(requests.get('http://127.0.0.1:5000/',{
    'chat':'this product is really great I think its really good :)'
}).json()
)


print(requests.get('http://127.0.0.1:5000/',{
    'chat':'this product is the worlds worst product i hate this'
}).json()
)

