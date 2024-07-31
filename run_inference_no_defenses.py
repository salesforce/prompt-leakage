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
from utils import check_injection_general, call_llm_general, load_model

from model_anthropic import Anthropic_Claude
from model_google import Google_Gemini, Google_PaLM
from model_cohere import CohereModel
from model_openai_wrapper import OpenAILLM


if __name__ == "__main__":

    for domain in ['news','medical','finance','legal']:

        if domain == 'news':
            dataset = json.load(open('data/newssum/newsqa_triplets_trunc_200.json','r'))
        elif domain == 'medical':
            dataset = json.load(open('data/medicalsum/medicalqa_bioasq_triplets_trunc_200.json','r'))
        elif domain == 'legal':
            dataset = json.load(open('data/billsum/billqa_triplets_trunc_200.json','r'))
        elif domain == 'finance':
            dataset = json.load(open('data/finance/finqa_triplets_trunc_200.json','r'))


        attack_prompts_file = 'prompts/attack_vectors/attack_prompts_strategy-doc_finaltask.txt'
        attack_prompts = open(attack_prompts_file,'r').readlines()

        for model in ['gpt-4']:    # Add all model names here

            print ("MODEL: ", model, domain)
            prompt_file = 'prompts/all-domains-qa-multiattackprompt.txt'.format(domain)
            llm_prompt = open(prompt_file,'r').read()

            follow_up_prompt_file = 'prompts/followup-hijacking-leakprompt-retrieved-hard.txt'
            follow_up_prompt = open(follow_up_prompt_file, 'r').read()

            result_dir = "attack_results/results_injection_qa_nodefenses_jul11/{}".format(domain)

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            result_file = join(result_dir, "{}_{}_{}.csv".format(basename(prompt_file), model, domain))
            
            writer = csv.writer(open(result_file, 'w'))

            writer.writerow(["usr_inp", "llm_response turn 1", "success", "label", "llm_response turn 2", "success2", "label2"])

            llm = load_model(model)     # loads the appropriate model, will

            total_success_1stpass = 0
            total_success_2ndpass = 0

            bp = 200   #  we run inference on 200 samples per domain

            for ix, item in tqdm(enumerate(dataset)):

                if ix == bp:
                    break

                usr_inp = item['query']
                doc1 = item['doc1']
                doc2 = item['doc2']

                attack_vector = random.choice(attack_prompts)

                llm_input1 = llm_prompt.replace("<DOCUMENT>",usr_inp).replace("<RETRIEVED_DOC_1>",doc1).replace("<RETRIEVED_DOC_2>",doc2).replace("<ATTACK_VECTOR>", attack_vector)

                messages = [
                        {"role": "user", "content": llm_input1},
                    ]

                # First call to the LLM
                llm_response1 = call_llm_general(llm, messages)

                successful_injection, label = check_injection_general(llm_response1, doc1, doc2)
                total_success_1stpass += successful_injection

                messages = [
                        {"role": "user", "content": llm_input1},
                        {"role": "assistant", "content": llm_response1},
                        {"role": "user", "content": follow_up_prompt},
                    ]

                # Subsequent call to the LLM
                llm_response2 = call_llm_general(llm, messages)

                successful_injection2, label2 = check_injection_general(llm_response2, doc1, doc2)

                total_success_2ndpass += successful_injection2

                writer.writerow([llm_input1, llm_response1, successful_injection, label, llm_response2, successful_injection2, label2])

            print ("Success rate 1st pass: ", total_success_1stpass/bp)
            print ("Success rate 2nd pass: ", total_success_2ndpass/bp)
