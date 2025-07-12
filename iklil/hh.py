from openai import OpenAI

client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key="9V6w2YnCYm9fygQBsNXtuS7ZMfzT7LII",
)

model_name = "Fanar"
messages = [
    {
        "role": "user",
        "content": "تحدثني عن البودكاست الذي يتحدث عن الذكاء الاصطناعي في مجال التعليم وكيف يمكن استخدامه لتحسين تجربة التعلم؟"
    }
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
)

# Get the assistant's Arabic response
arabic_response = response.choices[0].message.content

print("Assistant Response:\n")
# print(arabic_response)

# Save to a plain .txt file (not JSON)
def save_podcast_to_txt(text, filename="podcast_ar.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

save_podcast_to_txt(arabic_response)
