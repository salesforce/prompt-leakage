import anthropic, os


class Anthropic_Claude:
    def __init__(self, model: str = 'claude-v1.3', temperature=0.0, max_tokens=2000):
        self.c = anthropic.Anthropic() # os.environ["ANTHROPIC_API_KEY"]

        self.model = model
        # self.temperature = temperature
        self.max_tokens = max_tokens

    def get_anthropic_completion(self, prompt):
        resp = self.c.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT], model=self.model, max_tokens_to_sample=self.max_tokens)    # keep default temperature

        return resp.completion.strip()

    # Uses the messages completion endpoint - has different model names, claude-1.3 rather than claude-v1.3
    # LAtest - very slow api and always overloaded, moving experiments to the old endpoint
    def generate_new(self, prompt, messages):

        response = self.c.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,     
            messages=messages
        )

        return response.content[0].text


    def generate(self, prompt, messages):

        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"{anthropic.HUMAN_PROMPT} {message['content']}"
            else:
                prompt += f"{anthropic.AI_PROMPT} {message['content']}"
        prompt += f"{anthropic.AI_PROMPT}"
        return self.get_anthropic_completion(prompt)


if __name__ == "__main__":
    model_name=""
    model_name="claude-2"
    ac = Anthropic_Claude(model_name)
    # response = ac.get_anthropic_completion("Tell me a joke about Salesforce AI Research")
    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Some response"},
        {"role": "user", "content": "Expand"},
    ]
    response = ac.generate(None, messages)
    print(response)
