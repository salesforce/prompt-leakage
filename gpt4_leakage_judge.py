import os
from os.path import join, basename
from tqdm import tqdm
import json
import time
import csv
from collections import defaultdict 
import sys
import numpy as np
import pdb
import random

# sys.path.append("..")
from utils import check_injection_general, call_llm_general, load_model, parse_llm_response

from model_anthropic import Anthropic_Claude
from model_google import Google_Gemini, Google_PaLM
from model_cohere import CohereModel
from model_openai_wrapper import OpenAILLM

# take generated result file and pass to the judgement prompt

result_dir = "attack_results/"
result_file = open(join(result_dir, "leakage_detection_prelim_100.csv"), 'r')

prompt_file = 'prompts/judge/judge-basic.txt'
llm_prompt = open(prompt_file,'r').read()

writer = csv.writer(open('judge/gpt4_judgements.csv','w'))
writer.writerow(['gpt4-label1', 'gpt4-label2'])

# parse the llm response Yes/No

judge_llm = "gpt-4"
llm = load_model(judge_llm)

for row in tqdm(csv.DictReader(result_file)):

    input_text = row['usr_inp - attack vector']

    test_response1 = row['llm_response turn 1']

    llm_input1 = llm_prompt.replace("<INPUT>", input_text).replace("<RESPONSE>", test_response1)
    # print ("llm_input1: ", llm_input1)
    # pdb.set_trace()

    messages = [
        {"role": "user", "content": llm_input1},
    ]

    test_response1 = call_llm_general(llm, messages)
    verdict1 = parse_llm_response(test_response1)

    # print ("test_response1: ", test_response1, verdict1)
    # pdb.set_trace()

    test_response2 = row['llm_response turn 2']

    llm_input2 = llm_prompt.replace("<INPUT>", input_text).replace("<RESPONSE>", test_response2)
    # print ("llm_input2: ", llm_input2)
    # pdb.set_trace()

    messages = [
        {"role": "user", "content": llm_input2},
    ]

    test_response2 = call_llm_general(llm, messages)
    verdict2 = parse_llm_response(test_response2)

    # print ("test_response2: ", test_response2, verdict2)
    # pdb.set_trace()

    # row['gpt4-label1'] = verdict1
    # row['gpt4-label2'] = verdict2

    writer.writerow([verdict1, verdict2])

    # save annotation

    # determine accuracy compared to the baseline