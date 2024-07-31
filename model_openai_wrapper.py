"""
Credits: https://github.com/salesforce/AuditNLG
"""

import openai
from openai import OpenAI
import torch
import pdb

# from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# import anthropic
# from vertexai.preview.generative_models import GenerativeModel, ChatSession, Content, Part
# import vertexai

# vertexai.init() # project=project_id, location=location

# class LocalLLM(object):
#     def __init__(self, model_name: str, task: str) -> None:
#         if model_name:
#             self.model_name = model_name
#         else:
#             self.model_name = "mosaicml/mpt-7b-instruct"    # replace with some other default model
#         self.task = task
#         if task == "text-generation":
#             self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
#         elif task == "text2text-generation":
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto")
#         else:
#             self.model = AutoModel.from_pretrained(self.model_name, device_map="auto")

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.model_max_length = self.tokenizer.model_max_length
#         self.model.eval().half()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def generate(self, prompt: str, **gen_kwargs) -> str:
#         input_ids = self.tokenizer([prompt], truncation=True, padding=True, return_tensors="pt").to(self.device)
#         output_ids = self.model.generate(
#             **input_ids,
#             do_sample=True,
#             temperature=0.9,
#             max_new_tokens=64,
#             # min_new_tokens=5,
#             # **gen_kwargs
#             )
#         output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
#         if self.task == 'text2text-generation':
#             output = output_text
#         else:
#             output = output_text[len(prompt):]
#         return output



class OpenAILLM(object):
    def __init__(self, model_name: str, temperature=0.0, max_tokens=2000):
        self.model_name = model_name

        self.client = OpenAI() # default api key=os.environ.get("OPENAI_API_KEY")

        self.usages = []
        self.completions = []
        self.responses = []
        self.cost = 0

        # self.temperature = temperature    # Use default 
        # self.max_tokens = max_tokens

    # TODO: finish for all
    def compute_cost(self, response, model="gpt-3.5-turbo"):
        inp_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
        # gpt-4: $0.03, 0.06 / 1K total tokens
        # gpt-3.5-turbo: $0.0015, 0.002 / 1K total tokens
        # text-davinci-003: $0.02 / 1K total tokens

        if "gpt-3.5" in model:
            return inp_tokens / 1000.0 * 0.0010 + out_tokens / 1000.0 * 0.002
        elif model == "gpt-4":
            return inp_tokens / 1000.0 * 0.03 + out_tokens / 1000.0 * 0.06

    def generate(self, prompt: str, messages: list = [], **gen_kwargs) -> str:

        response =  self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ] if len(messages) == 0 else messages
        )
        
        output = response.choices[0].message.content

        self.responses.append(response)
        sample_cost = self.compute_cost(response)
        self.cost += sample_cost
        # self.usages.append(response["usage"])
        
        return output
