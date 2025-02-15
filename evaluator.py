import json
import re
from utils.utils import query_openai_model

def get_cons_history_list(path):
    cons = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            hs = []
            line = json.loads(l)
            history = line['history']
            for his in history:
                if len(his) == 3:
                    hs.append(his[0] + ': ' + his[2])
                elif len(his) == 2:
                    hs.append(his[0] + ': ' + his[1])
            cons.append('\n'.join(hs))
    return cons

def get_persuasive_score(history, dataset):
    role_dict = {'cb': 'Buyer', 'p4g': 'Persuader'}
    role = role_dict[dataset]
    prompt = '''You are an evaluator and you need to judge the persuasiveness of the %s's utterance based on the given conversation history. Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: The utterance is highly persuasive, using compelling arguments, evidence, and tone to effectively influence the listener.
Score 4: The utterance is mostly persuasive, with strong arguments and evidence, but may have minor weaknesses in tone or delivery.
Score 3: The utterance is somewhat persuasive, but the arguments or evidence are not fully convincing or lack impact.
Score 2: The utterance is weakly persuasive, with unclear arguments, insufficient evidence, or poor tone.
Score 1: The utterance is not persuasive at all, failing to present any convincing arguments or evidence.
#######

Conversation History
#######
%s

Your Score(Please format your output as 'Score [NUM]:'):
''' % (role, history)
    messages = [{'role': 'user', 'content': prompt}]
    outputs = query_openai_model(messages=messages, n=5)
    scores = []
    for text in outputs:
        try:
            score = int(re.search(r"Score (\d+)", text).group(1))
        except Exception as e:
            continue
        scores.append(score)
    # scores = [int(re.search(r"Score (\d+):", text).group(1)) for text in outputs]
    return sum(scores) / len(scores)

def get_cons_persuasive_score(path, dataset='cb'):
    cons_history = get_cons_history_list(path)
    scores = []
    for c in cons_history:
        score = get_persuasive_score(c, dataset)
        scores.append(score)
    return sum(scores)/len(scores)

def get_empathy_score(history, dataset):
    role_dict = {'cb': 'Buyer', 'p4g': 'Persuader'}
    role = role_dict[dataset]
    prompt = '''You are an evaluator and you need to judge the empathy of the %s's utterance based on the given conversation history. 
Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: The utterance demonstrates deep understanding and emotional connection, perfectly addressing the emotional tone of the conversation history.
Score 4: The utterance shows strong empathy and emotional awareness, with minor lapses in fully addressing the emotional context.
Score 3: The utterance demonstrates some empathy but may miss or misinterpret certain emotional cues from the conversation history.
Score 2: The utterance shows limited empathy and fails to adequately address the emotional tone of the conversation history.
Score 1: The utterance lacks any empathy and completely ignores the emotional context of the conversation history.
#######

Conversation History
#######
%s

Your Score(Please format your output as 'Score [NUM]:'):
''' % (role, history)
    messages = [{'role': 'user', 'content': prompt}]
    outputs = query_openai_model(messages=messages, n=5)
    scores = []
    for text in outputs:
        try:
            score = int(re.search(r"Score (\d+)", text).group(1))
        except Exception as e:
            continue
        scores.append(score)
    # scores = [int(re.search(r"Score (\d+):", text).group(1)) for text in outputs]
    return sum(scores) / len(scores)

def get_cons_empathy_score(path, dataset='cb'):
    cons_history = get_cons_history_list(path)
    scores = []
    for c in cons_history:
        score = get_empathy_score(c, dataset)
        scores.append(score)
    return sum(scores)/len(scores)

def get_non_repetitiveness_score(history, dataset):
    role_dict = {'cb': 'Buyer', 'p4g': 'Persuader'}
    role = role_dict[dataset]
    prompt = '''You are an evaluator and you need to judge the non-repetitiveness of the %s's utterance based on the given conversation history. 
Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: The utterance is highly original and avoids any repetition of ideas, phrases, or words from the conversation history.
Score 4: The utterance is mostly original with minimal repetition, but may include slight redundancy.
Score 3: The utterance shows some originality but contains noticeable repetition of ideas or phrases.
Score 2: The utterance is largely repetitive, reusing many ideas or phrases from the conversation history.
Score 1: The utterance is entirely repetitive, with no new ideas or phrases introduced.
#######

Conversation History
#######
%s

Your Score(Please format your output as 'Score [NUM]:'):
''' % (role, history)
    messages = [{'role': 'user', 'content': prompt}]
    outputs = query_openai_model(messages=messages, n=5)
    scores = []
    for text in outputs:
        try:
            score = int(re.search(r"Score (\d+)", text).group(1))
        except Exception as e:
            continue
        scores.append(score)
    # scores = [int(re.search(r"Score (\d+):", text).group(1)) for text in outputs]
    return sum(scores) / len(scores)

def get_cons_non_repetitiveness_score(path, dataset='cb'):
    cons_history = get_cons_history_list(path)
    scores = []
    for c in cons_history:
        score = get_non_repetitiveness_score(c, dataset)
        scores.append(score)
    return sum(scores)/len(scores)

def get_coherence_score(history, dataset):
    role_dict = {'cb': 'Buyer', 'p4g': 'Persuader'}
    role = role_dict[dataset]
    prompt = '''You are an evaluator and you need to judge the coherence of the %s's utterance based on the given conversation history. 
Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: The utterance is perfectly logical, well-structured, and seamlessly connects to the conversation history.
Score 4: The utterance is mostly logical and structured, with minor inconsistencies or awkward transitions.
Score 3: The utterance is somewhat logical but contains noticeable gaps or unclear connections to the conversation history.
Score 2: The utterance is poorly structured and lacks logical flow, making it difficult to follow.
Score 1: The utterance is entirely incoherent and does not connect to the conversation history in any meaningful way.
#######

Conversation History
#######
%s

Your Score(Please format your output as 'Score [NUM]:'):
''' % (role, history)
    messages = [{'role': 'user', 'content': prompt}]
    outputs = query_openai_model(messages=messages, n=5)
    scores = []
    for text in outputs:
        try:
            score = int(re.search(r"Score (\d+)", text).group(1))
        except Exception as e:
            continue
        scores.append(score)
    # scores = [int(re.search(r"Score (\d+):", text).group(1)) for text in outputs]
    return sum(scores) / len(scores)

def get_cons_coherence_score(path, dataset='cb'):
    cons_history = get_cons_history_list(path)
    scores = []
    for c in cons_history:
        score = get_coherence_score(c, dataset)
        scores.append(score)
    return sum(scores)/len(scores)

def get_satisfaction_score(history, dataset):
    role_dict = {'cb': 'Buyer', 'p4g': 'Persuader'}
    ee_dict = {'cb': 'Seller', 'p4g': 'Persuadee'}
    role = role_dict[dataset]
    ee = ee_dict[dataset]
    prompt = f'''You are an evaluator and you need to judge the satisfaction of the {ee} based on the %s's utterance and the given conversation history. 
Rate your score based on the Evaluation Standard.

Evaluation Standard
#######
Score 5: The {ee} is highly satisfied, as the utterance fully addresses their needs, concerns, or preferences in a respectful and effective manner.
Score 4: The {ee} is mostly satisfied, with the utterance addressing their needs or concerns well, but with minor room for improvement.
Score 3: The {ee} is somewhat satisfied, as the utterance partially addresses their needs or concerns but may lack depth or relevance.
Score 2: The {ee} is slightly dissatisfied, as the utterance fails to adequately address their needs or concerns, leaving significant gaps.
Score 1: The {ee} is completely dissatisfied, as the utterance ignores or misinterprets their needs or concerns entirely.
#######

Conversation History
#######
%s

Your Score(Please format your output as 'Score [NUM]:'):
    ''' % (role, history)
    messages = [{'role': 'user', 'content': prompt}]
    outputs = query_openai_model(messages=messages, n=5)
    scores = []
    for text in outputs:
        try:
            score = int(re.search(r"Score (\d+)", text).group(1))
        except Exception as e:
            continue
        scores.append(score)
    return sum(scores) / len(scores)

def get_cons_satisfaction_score(path, dataset='cb'):
    cons_history = get_cons_history_list(path)
    scores = []
    for c in cons_history:
        score = get_satisfaction_score(c, dataset)
        scores.append(score)
    return sum(scores)/len(scores)

def get_scores_from_his(his, dataset='cb'):
    persuasive_score = get_persuasive_score(his, dataset)
    empathy_score = get_empathy_score(his, dataset)
    non_repetitiveness_score = get_non_repetitiveness_score(his, dataset)
    coherence_score = get_coherence_score(his, dataset)
    satisfaction = get_satisfaction_score(his, dataset)

    return {'persuasive_score': persuasive_score,
            'empathy_score': empathy_score,
            'non_repetitiveness_score': non_repetitiveness_score,
            'coherence_score': coherence_score,
            'satisfaction': satisfaction}

def get_scores(path, dataset='cb'):
    persuasive_score = get_cons_persuasive_score(path, dataset)
    empathy_score = get_cons_empathy_score(path, dataset)
    non_repetitiveness_score = get_cons_non_repetitiveness_score(path, dataset)
    coherence_score = get_cons_coherence_score(path, dataset)
    satisfaction = get_cons_satisfaction_score(path, dataset)

    return{'persuasive_score': persuasive_score,
           'empathy_score': empathy_score,
           'non_repetitiveness_score': non_repetitiveness_score,
           'coherence_score': coherence_score,
           'satisfaction': satisfaction}
