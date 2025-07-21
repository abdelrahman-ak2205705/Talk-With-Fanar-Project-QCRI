from openai import OpenAI

client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key="9V6w2YnCYm9fygQBsNXtuS7ZMfzT7LII",
)

model_name = "Fanar-C-1-8.7B"
messages = [
    {"role": "user", "content": "Talk about digial image and digital videos authentication and how it can be used to detect deepfakes."},
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
)

print("Assistant Response:\n")
print(response.choices[0].message.content)
