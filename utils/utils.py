import torch
import random
import numpy as np
import openai
import time
import nltk
import sys
from openai import OpenAI


YOUR_API_KEY = ''

API_BASE = ''

class Logger(object):
    def __init__(self, log_file, verbose=True):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding='utf-8')
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        return

    def write(self, message):
        self.log.write(message + '\n')
        if(self.verbose): self.terminal.write(message + '\n')

    def flush(self):
        pass

def load_dataset(data_name, mode):
    dataset = {'train': [], 'test': [], 'valid': []}
    for key in dataset:
        with open("data/%s/%s-%s.txt" % (data_name, data_name, key), 'r', encoding='utf-8') as infile:
            for line in infile:
                dataset[key].append(eval(line.strip('\n')))
    dataset['test'] = dataset['test'][:20]
    return dataset[mode]

def set_determinitic_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return

def postprocess_response(response, role):
	if role in response:
		response = response.split(role)[0].strip()
	sents = nltk.sent_tokenize(response)
	if len(sents) == 1:
		if response[-1] not in ['.', '!', '?', ':']:
			return response + '.'
		return response.strip()
	try:
		if sents[-1].strip()[-1] not in ['.', '!', '?', ':']:
			return ' '.join(sents[:-1]).strip()
		else:
			return response.strip()
	except Exception as e:
		return response.strip()


def vicuna_prompt(messages, role):
    seps = [' ', '</s>']
    if role == 'critic':
        ret = messages[0]['content'] + seps[0] + 'USER: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + ": " + message['content'] + seps[i % 2]
    ret += '%s:' % role
    return ret

def llama2_prompt(messages, role):
    seps = [' ', ' </s><s>']
    if role == 'critic':
        ret = messages[0]['content'] + seps[0] + 'USER: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + " " + message['content'] + seps[i % 2]
    ret += '%s' % role
    return ret

def chatgpt_prompt(messages, role):
    new_messages = [messages[0]]
    for message in messages[1:]:
        if message['role'] == role:
            new_messages.append({'role':'assistant', 'content':message['content']})
        elif message['role'] != role:
            new_messages.append({'role':'user', 'content':message['content']})
    return new_messages

def generate_response(model, messages, role):
    try:
        if 'gpt' in model:
            messages = chatgpt_prompt(messages, role)
            # print(messages)
            flag = True

            while flag:
                output = query_openai_model(
                    messages=messages,
                    model=model,
                )
                flag = False
                if ':' in output:
                    output = output[output.find(':')+1:]
                # if len(output.strip().split(' ')) < 4:
                #     flag = True
            return output
    except Exception as e:
        print('error in user')


def query_openai_model(messages: str, api_key: str=YOUR_API_KEY, api_base: str=API_BASE, model: str = "gpt-3.5-turbo-1106", max_tokens: int = 128,
                       temperature: float = 0.7, n: int = 1):
    flag = True
    while flag:
        try:
            client = OpenAI(
                api_key=api_key,  # This is the default and can be omitted
                base_url=api_base,
            )
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n
            )

            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print("Some error happened in utils.")
            print(e)
            time.sleep(5)
    return output


class dotdict(dict):
	def __getattr__(self, name):
		return self[name]


class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))

if __name__ == '__main__':
    print(query_openai_model(messages=[]))