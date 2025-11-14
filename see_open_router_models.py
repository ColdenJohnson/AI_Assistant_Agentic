import requests
url = "https://openrouter.ai/api/v1/models"
headers = {"Authorization": "Bearer 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"}
response = requests.get(url, headers=headers)
# print(response.json())

models = response.json().get("data", [])
claude_models = [m for m in models if "grok" in m["id"].lower()]

for m in claude_models:
    print(m["id"], "-", m.get("name"))





# from openai import OpenAI

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key="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",
# )

# response = client.get("/models")

# # If the library doesn’t support raw get, do this alternative:
# # import requests
# # response = requests.get(
# #     "https://openrouter.ai/api/v1/models",
# #     headers={"Authorization": f"Bearer {YOUR_API_KEY_HERE}"}
# # )

# models = response.json().get("data", [])
# for m in models:
#     print(m["id"], "–", m.get("name"))
