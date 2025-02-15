import openai
import time
from openai import OpenAI
import copy
import re
import random
from reflexion.prompts import m_cb_observator_prompt, m_p4g_observator_prompt, CB_REFLECTION_DRAFT_INSTRUCTION, PG_REFLECTION_DRAFT_INSTRUCTION



def get_history(state, n=3):  # 3
    if len(state.history) >= n:
        history_list = state.history[-n:]
    else:
        history_list = state.history
    history = ''
    for it in history_list:
        history += it[0] + ':' + it[2] + '\n'
    return history.strip()


class Moderator:
    def __init__(self):
        pass
    def check_end(self, debate_state):
        history = ''
        for d in debate_state[-2:]:
            history += d[0] + ':' +d[2]

        messages = [{'role': 'system', 'content': 'You are a moderator in a debate. '
                                                  'Your task is to determine whether the two debaters have reached a agreement or have expressed the same opinion. '
                                                  'You can only reply with one of the following sentences: They have not reached. They have reached. '
                                                  '\nConversation:%s' % history}]
        outputs = query_openai_model(messages, n=3)
        have_reached, have_not_reached = 0, 0
        for o in outputs:
            if 'have reached' in o.lower():
                have_reached += 1
            elif 'have not reached' in o.lower():
                have_not_reached += 1
        return have_not_reached > have_reached

    def get_summery(self, debate_state):
        history = ''
        for d in debate_state:
            history += d[0] + ':' + d[2]

        messages = [{'role': 'system', 'content': 'You are the moderator of a negotiation task, and your task is to summarize the debate process between the two debaters and provide a final conclusion in short words..'
                                                  '\nConversation:%s' % history}]
        output = query_openai_model(messages)
        return output


class ExpReflexion:
    def __init__(self, state, dataset):
        self.state = state
        self.memory = []
        self.future = ''
        self.M_OBJDICT = {"p4g": m_p4g_observator_prompt, "cb": m_cb_observator_prompt}
        self.CRAFTDICT = {'p4g': PG_REFLECTION_DRAFT_INSTRUCTION, 'cb': CB_REFLECTION_DRAFT_INSTRUCTION}
        self.SYS_ROLE_DICT = {'p4g': 'Persuader', 'cb': 'Buyer'}
        self.USR_ROLE_DICT = {'p4g': 'Persuadee', 'cb': 'Seller'}
        self.HIS_LEN_DICT = {'p4g': 3, 'cb': 20}
        self.dataset = dataset
        if self.dataset == 'cb':
            self.user_max_price = state.buyer_price

    def get_observation(self, dataset, his_len):
        # expression whether user center
        obj_prompt = self.OBJDICT[dataset]
        messages = [{
            'role': 'user',
            'content': obj_prompt.format(history=self.get_history(his_len))
        }]
        obersation = query_openai_model(messages, n=5)
        return obersation

    def get_muti_observation(self, dataset, his_len, critics_num=5):
        muti_obj_prompt = self.M_OBJDICT[dataset]
        messages = [{
            'role': 'user',
            'content': muti_obj_prompt.format(history=self.get_history(his_len))
        }]
        obersation = query_openai_model(messages, n=critics_num)
        return obersation

    def observation_check(self, obs, dataset):
        not_done, done = 0, 0
        for o in obs:
            if 'have not done' in o:
                not_done += 1
            elif 'have done' in o:
                done += 1
        return done >= not_done

    def muti_observation_check(self, obs, dataset):
        y_num, n_num = 0, 0
        for o in obs:
            if 'yes' in o.lower()[:12]:
                y_num += 1
            elif 'no' in o.lower()[:12]:
                n_num += 1
        if dataset == 'cb':
            return y_num > n_num
        if dataset == 'p4g':
            return n_num == 0

    def reflections2craft(self, refs, dataset):
        draft_prompt = self.CRAFTDICT[dataset]
        no_refs = []
        for r in refs:
            if 'no' in r.lower()[:12]:
                r_i = r.lower().find('suggestion')
                ref = r[r_i + len('suggestion'):]
                no_refs.append(ref)
        messages = [{
            'role': 'user',
            'content': draft_prompt.format(num=len(refs), reflections='\n'.join(no_refs))
        }]
        craft = query_openai_model(messages, n=3)
        return craft

    def single_reflections2craft(self, refs, dataset):
        no_refs = []
        for r in refs:
            if 'no' in r.lower()[:12]:
                r_i = r.lower().find('suggestion')
                ref = r[r_i + len('suggestion'):]
                no_refs.append(ref)

        return random.choice(no_refs)


    def actor(self, ref, reflexion_cur_num, dataset):
        from utils.character_utils import pg_sys_ans, cb_sys_ans
        bad_res = []
        for mem in self.memory:
            bad_res.append([mem[0][-1], mem[1]])
        charactor = ''
        for br in bad_res:
            ref = br[0][0] + ': ' + br[0][2] + '\n' + 'reflection: ' + br[1] + '\n'
            charactor = br[0][0]
        self.state.history.pop()
        if dataset == 'p4g':
            stategy, reaction = pg_sys_ans(state=self.state, reactor=True, reflexion=ref, future=self.future, reflexion_cur_num=reflexion_cur_num)
        elif dataset == 'cb':
            stategy, reaction = cb_sys_ans(state=self.state, reactor=True, reflexion=ref, future=self.future, reflexion_cur_num=reflexion_cur_num)
        self.state.add_single(charactor, stategy, reaction)
        return (reaction)

    def save2memory(self, ref):
        # save the fail act to memory
        self.memory.append([copy.deepcopy(self.state).history, ref])

    def reset4memory(self):
        self.memory = []

    def muti_reflexion(self, logger, dataset='cb', max_reflexion=7, critics_num=5, single_critic=False):
        his_len = self.HIS_LEN_DICT[dataset]
        muti_sub_reflection = self.get_muti_observation(dataset, his_len, critics_num)
        output_observation = '\n'.join(muti_sub_reflection)
        if_thought = self.muti_observation_check(muti_sub_reflection, dataset)
        logger.write(f'obersation start:--------------------')
        logger.write(output_observation)
        logger.write(f'obersation end:--------------------')
        reflexion_cur_num = 0
        while not if_thought and reflexion_cur_num < max_reflexion:
            reflexion_cur_num += 1
            if not single_critic:
                reflection_craft = self.reflections2craft(muti_sub_reflection, dataset)
            else:
                reflection_craft = self.single_reflections2craft(muti_sub_reflection, dataset)
            logger.write(f'craft start:--------------------')
            logger.write(reflection_craft[0])
            logger.write(f'craft end:--------------------')
            self.save2memory(reflection_craft[0])

            output = self.actor(reflection_craft[0], reflexion_cur_num, dataset)
            logger.write(f'actor-{reflexion_cur_num}: {output}')

            muti_sub_reflection = self.get_muti_observation(dataset, his_len)
            if_thought = self.muti_observation_check(muti_sub_reflection, dataset)
            logger.write(f"{self.SYS_ROLE_DICT[dataset]}: ({self.state.history[-1][1]}){self.state.history[-1][2]}")

            logger.write(f'obersation start:--------------------')
            logger.write(output_observation)
            logger.write(f'obersation end:--------------------')

        if reflexion_cur_num > 0:
            return copy.deepcopy(self.memory), reflexion_cur_num

        return None, reflexion_cur_num



    def reflexion(self, logger, dataset='p4g', max_reflexion=7):
        his_len = self.HIS_LEN_DICT[dataset]
        obersation = self.get_observation(dataset, his_len)
        logger.write(f'obersation: {obersation}')
        if_thought = self.observation_check(obersation, dataset)
        reflexion_cur_num = 0

        while not if_thought and reflexion_cur_num < max_reflexion:
            reflexion_cur_num += 1
            reflection = self.thought(dataset, his_len)
            self.save2memory(reflection)
            logger.write(f'reflection: {reflection}')
            output = self.actor(reflection, reflexion_cur_num, dataset)

            logger.write(f'actor-{reflexion_cur_num}: {output}')

            obersation = self.get_observation(dataset, his_len)
            logger.write(f'obersation: {obersation}')
            if_thought = self.observation_check(obersation, dataset)

            logger.write(f"{self.SYS_ROLE_DICT[dataset]}: ({self.state.history[-1][1]}){self.state.history[-1][2]}")

        if reflexion_cur_num > 0:
            return copy.deepcopy(self.memory), reflexion_cur_num

        return None, reflexion_cur_num

    def get_history(self, n=3):
        if len(self.state.history) >= n:
            history_list = self.state.history[-n:]
        else:
            history_list = self.state.history
        history = ''
        for it in history_list:
            history += it[0] + ':' + it[2] + '\n'
        return history.strip()


YOUR_API_KEY = ''
API_BASE = ""

def query_openai_model(messages: str, api_key: str = YOUR_API_KEY, api_base: str = API_BASE,
                       model: str = "gpt-3.5-turbo", max_tokens: int = 128,
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
            print("Some error happened here.")
            print(e)
            time.sleep(5)
    return output


