from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
