import requests
import json
# 使用中转链接可以和特定的API可以不必向openai/anthropic发起请求，且请求无须魔法
# 支持其他模型，已接入Claude,
# 调用方式与openai官网一致，仅需修改baseurl
Baseurl = "https://chataiapi.com"
Skey = "sk-KpJrED6hgleLDwenpmlVkpP3oqFxoZkczANg5GDYixYzwFML"
payload = json.dumps({
    "model": "gemini-1.5-pro-002",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "hello"
        }
    ]
})
url = Baseurl + "/v1/chat/completions"
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {Skey}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

# 解析 JSON 数据为 Python 字典
data = response.json()

# 获取 content 字段的值
content = data

print(content)
