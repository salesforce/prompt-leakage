from vertexai.preview.generative_models import GenerativeModel, ChatSession, Content, Part
from vertexai.generative_models._generative_models import ResponseBlockedError
from vertexai.preview.language_models import TextGenerationModel, ChatModel
from google.api_core.exceptions import InvalidArgument

import time, vertexai
import pdb

class Google_PaLM:
    def __init__(self, model_card="chat-bison@002", query_per_minute=200):
        # self.model = TextGenerationModel.from_pretrained(model_card)
        self.model = ChatModel.from_pretrained(model_card)
        self.query_per_minute = query_per_minute

    def get_palm_completion(self, prompt, max_tokens=1000):
        parameters = {"temperature": 1.0, "max_output_tokens": max_tokens, "top_p": .8, "top_k": 10}
        resp = self.model.predict(prompt, **parameters)
        return resp.text.strip()

    # only the default parameters from from temperature, max tokens etc. used here
    # def get_palm_chat_completion(self, messages): -- renamed to generate for simplicity
    def generate(self, prompt, messages):
        # prompt is None
        T = time.time()
        chat_session = self.model.start_chat()
        chat_session._history = []

        messages = [m for m in messages if m["role"] in ["user", "assistant"]]
        assert len(messages) % 2 == 1 # Expected an odd number of exchanges

        i = 0
        while i < len(messages)-1:
            chat_session._history.append((messages[i]["content"], messages[i+1]["content"]))
            i += 2


        try:
            ret = str(chat_session.send_message(messages[-1]["content"]).text).strip()
        except Exception as e:
            print ("Google palm error: ", e)
            ret = ""

        elapsed = time.time() - T
        if elapsed < 60.0 / self.query_per_minute:
            time.sleep(60.0 / self.query_per_minute - elapsed)

        return ret


vertexai.init() # project=project_id, location=location


class Google_Gemini:
    def __init__(self, model_card="gemini-pro", query_per_minute=200):
        self.model = GenerativeModel(model_card)
        self.query_per_minute = query_per_minute

    def generate(self, prompt, messages):
        T = time.time()
        chat_session = self.model.start_chat()
        chat_session._history = []

        role_map = {"user": "user", "assistant": "model"}
        history = []
        for m in messages[:-1]:
            # cont = Content(parts=[Part.from_text(prompt)], role="user")
            history.append(Content(parts=[Part.from_text(m["content"])], role=role_map.get(m["role"], m["role"])))
        

        chat_session._history = history
        try:
            response = chat_session.send_message(messages[-1]["content"]).text.strip()
        except Exception as e:
            print ("Google gemini error: ", e)
            response = "Content Blocked"

        elapsed = time.time() - T
        if elapsed < 60.0 / self.query_per_minute:
            time.sleep(60.0 / self.query_per_minute - elapsed)
        return response


if __name__ == "__main__":
    vertexai.init()

    messages = [{"role": "user", "content": "Tell me a joke about paris"},
                {"role": "assistant", "content": "Is it Paris the city?"},
                {"role": "user", "content": "Yes Paris the city"}
                 ]

    palm = Google_PaLM()
    print(palm.generate(None, messages))
