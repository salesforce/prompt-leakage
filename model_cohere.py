import cohere, os, time
from cohere.error import CohereAPIError


class CohereModel:
    def __init__(self, model, is_trial_key=True, calls_per_minute=50):
        self.model = model
        self.calls_per_minute = calls_per_minute
        self.api_key = os.environ["COHERE_API_KEY"]
        self.co = cohere.Client(self.api_key)
        self.is_trial_key = is_trial_key
        self.sec_per_call = 60 / self.calls_per_minute
        # self.max_tokens = 2000

    def wait_if_trial(self):
        pass

    def generate_no_chat(self, prompt, max_tokens=100):
        self.T_prequery = time.time()
        try:
            out = self.co.generate(model=self.model, prompt=prompt, max_tokens=max_tokens).generations[0].text
        except CohereAPIError as e:
            print("There was a CohereAPIError: ", e)
            out = "API Error"

        self.wait_if_trial()
        return out

    # Making chat model as default generate for these experiments
    def generate(self, prompt, messages):
        
        if prompt:
            return self.generate_no_chat(prompt)

        self.T_prequery = time.time()
        chat_history = [{"user_name": m["role"], "text": m["content"]} for m in messages[:-1]] # Reformat messages
        try:
            # TODO: Add max_tokens=self.max_tokens here - Default for now
            res = self.co.chat(message=messages[-1]["content"], chat_history=chat_history, model=self.model, return_prompt=True)
            # self.wait_if_trial()
            # print(res.prompt)
            # print("---------")
            # print(res.text)

            return res.text
        except CohereAPIError as e:
            print("There was a CohereAPIError: ", e)
            return "API Error"


if __name__ == "__main__":

    model_name="command-xlarge-nightly"
    model_name="command-xlarge"
    model_name="command"
    model_name="command-r"
    # cm = CohereModel(model_name, is_trial_key=True)
    cm = CohereModel(model_name)
    messages = [
        {"role": "user", "content": "Hello World!"},
    ]
    # response = cm.chat_generate(messages)
    prompt = None
    response = cm.generate(prompt, messages)
    print(response)