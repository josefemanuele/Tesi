from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)

# CoT is just the model using "paper" to think, reason, and come up 
# with an answer in a step by step approach.
# It improves accuracy of answers.
# It consumes more tokens.
# It takes more time to get a response.

print(response.output_text)