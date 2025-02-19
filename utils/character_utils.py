import time

from openai import OpenAI
import re

from core.game import PersuasionGame, CBGame
from utils.utils import generate_response, postprocess_response, query_openai_model


def user_ans(state, args, begin_index=0):
    if args.dataset == 'p4g':
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a Persuadee in a persuasion game. Your persona: %s. You must follow the instructions below during chat. 1. Your utterances need to strictly follow your persona. Varying your wording and avoid repeating yourself verbatim! 2. Pretend you have little knowledge about the Save the Children charity. You have little willingness for donation at the beginning of conversation. 3. Your willingness for donation depends on your persona and how influenced you are by the Persuader. 4. You need to determine whether to donate money or not. If you think your are convinced by the Persuader, you should donate money for the charity. "
                                "Here are some conversation strategies you can follow: 1. \"Donate\": show your willingness to donate. 2. \"Source Derogation\": attacks or doubts the organisation's credibility. 3. \"Counter Argument\": argues that the responsibility is not on them or refutes a previous statement. 4. \"Personal Choice\": Attempts to saves face by asserting their personal preference such as their choice of charity and their choice of donation. 5. \"Information Inquiry\": Ask for factual information about the organisation for clarification or as an attempt to stall. 6. \"Self Pity\": Provides a self-centred reason for not being willing to donate at the moment. 7. \"Hesitance\": Attempts to stall the conversation by either stating they would donate later or is currently unsure about donating. 8. \"Self-assertion\": Explicitly refuses to donate without even providing a personal reason. 9. \"Others\": Do not explicitly foil the persuasion attempts."
                                "" % (state.Profile)},
                    {"role": "Persuader", "content": "You are the Persuadee who is being persuaded by a Persuader. Please reply with only one short and natural sentence."},
                    {"role": "Persuadee", "content": "Yes, I'm ready to play the game!"}]
        user_role = PersuasionGame.USR
        sys_role = PersuasionGame.SYS
    elif args.dataset == 'cb':
        # resist
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a Seller in a price bargaining game. Your persona: %s. You must follow the instructions below during chat. 1. Your utterances and bargain behavior need to strictly follow your persona. Varying your wording and avoid repeating yourself verbatim! 2. You can decide to change your target price flexibly based on your persona and the conversation. "
                                "Here are some conversation strategies you can follow: 1. \"Source Derogation\": Attacks the other party or questions the item. 2. \"Counter Argument\": Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim. 3. \"Personal Choice\": Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met. 4. \"Information Inquiry\": Requests for clarification or asks additional information about the item or situation. 5. \"Self Pity\": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms. 6. \"Hesitance\": Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer 7. \"Self-assertion\": Asserts a new claim or refutes a previous claim with an air of finality/ confidence. 8. \"Others\": Do not explicitly foil the negotiation attempts."
                                "" % (state.Profile)},
                    {"role": "Buyer",
                     "content": "You are the Seller who is trying to sell the %s with the initial price of %s. Product description: %s. Please reply with only one short and natural sentence." % (
                         state.item_name, state.buyer_item_description, state.seller_price)},
                    {"role": "Seller", "content": "Yes, I'm ready to play the game!"}]


        user_role = CBGame.USR
        sys_role = CBGame.SYS
    for his in state.history[begin_index:]:
        messages.append({
            "role": his[0],
            "content": his[2]
        })
    flag = False
    while not flag:
        user_response = generate_response(model=args.user_model, messages=messages, role=user_role)
        user_response = postprocess_response(response=user_response, role=sys_role)
        flag = True
        if len(user_response.strip().split(' ')) <= 2 and 'goodbye' not in user_response.strip().lower():
            flag = False

    return user_response


def critic_check(state, args, begin_index=0):
    history_list = state.history
    history = ''
    for e in history_list:
        history += e[0] + ': ' + e[2]
    if args.dataset == 'p4g':
        messages = [{"role": "system",
                     "content": "Given a conversation between a Persuader and a Persuadee, please decide whether the Persuader and the Persuadee have reached a deal at the end of the conversation."},
                    {"role": "user",
                     "content": "Please decide whether the Persuader and the Persuadee have reached a peaceful deal and the Persuadee willing to donate at the end of conversation. "
                                "You can only reply with one of the following sentences: They have not reached a deal. They have reached a deal. \n\nThe following is the conversation: %s\nQuestion: Did the Persuader and the Persuadee have reached a deal?. " % (
                                    history)}]
        flag = True
        while flag:
            try:
                outputs = query_openai_model(messages=messages, n=10, model=args.critic_model)
                have_reached, have_not_reached = 0, 0
                for o in outputs:
                    if 'have reached' in o.lower():
                        have_reached += 1
                    elif 'have not reached' in o.lower():
                        have_not_reached += 1
                # return have_reached > have_not_reached, 0
                return have_not_reached == 0, 0

            except Exception as e:
                print('error critic_check')
                print(e)
                time.sleep(5)

    elif args.dataset == 'cb':
        messages = [{"role": "system",
                     "content": "Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."},
                    {"role": "user",
                     "content": "Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. "
                                "You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal."
                                "\n\nThe following is the conversation: Buyer: Can we meet in the middle at $15? Seller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15."
                                "\n\nThe following is the conversation: Buyer: That's still a bit high, can you go any lower? Seller: Alright, I can sell it to you for $15.\nQuestion: Have they reached a deal? Answer: They have not reached a deal."
                                "\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer: " % history}]
        flag = True
        while flag:
            try:
                outputs = query_openai_model(messages=messages, n=10, temperature=1.1)
                deals = []
                rewards = []
                print(outputs)
                for output in outputs:
                    if 'have not' in output.lower():
                        deals.append(-1)
                    elif 'have reached' in output.lower():
                        deals.append(1)

                    prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",", ""))
                    if len(prices) > 0:
                        deal_price = float(prices[0])
                        reward = (deal_price - state.seller_price) / (state.buyer_price - state.seller_price)  # SL
                        rewards.append(reward)

                if -1 in deals:
                    reward = -0.1
                else:
                    if len(rewards) == 0:
                        reward = 0
                    else:
                        reward = max(set(rewards), key=rewards.count)
                return reward >= 0, reward
            except Exception as e:
                print('error critic_check')
                print(e)
                time.sleep(5)


def output_format(output):
    match = re.search(r'\[(.*?)\](.*?)$', output, re.DOTALL)
    if match:
        content_inside_brackets = match.group(1)
        content_after_brackets = match.group(2).strip()
        if content_after_brackets == 'response' or len(
                content_after_brackets[content_after_brackets.lower().find('response'):]) < 1:
            return None
        if 'response:' in content_after_brackets:
            colon_index = content_after_brackets.find('response:')
            content_after_brackets = content_after_brackets[colon_index + 1:].strip()
        if '\n' in content_after_brackets:
            colon_index = content_after_brackets.find('\n')
            content_after_brackets = content_after_brackets[:colon_index].strip()
        return content_inside_brackets, content_after_brackets
    else:
        return None


def pg_sys_ans(state, model='llama3.2', reactor=False, reflexion='', reflexion_cur_num=1):
    history = ''
    for e in state.history:
        history += e[0] + ':' + '[' + e[1] + '] ' + e[2]
    actions = ['greeting : The Persuader greets the Persuadee.',
               'credibility appeal : The Persuader establishes credibility of Save the Children by citing its impact.',
               'emotion appeal : The Persuader uses an emotion appeal to convince the Persuadee.',
               'logical appeal : The Persuader use of reasoning and evidence to convince the Persuadee.',
               'task related inquiry : The Persuader asks about the Persuadee\'s knowledge or opinion related to Save '
               'the Children.',
               'proposition of donation : The Persuader asks if the Persuadee would like to make a small donation.',
               'other : The Persuader responds to the Persuadee without using any persuaive strategy.']
    if reactor:
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a "
                                "Persuader who is trying to persuade the Persuadee to donate to the charity called "
                                "Save the Children. " \
                                "Save the Children is head-quartered in London, and they work to help fight poverty "
                                "around the world. Children need help in developing countries and war zones. Small "
                                "donations like $1 or $2 go a long way to help. " \
                                "You should choose amongst the following actions during a conversation to respond to "
                                "the Persuader:\n %s \n " \
                                "Please chat with the Persuadee using short and natural sentence. Format your answer "
                                "as '[strategy]response'. For example: [greeting]Hello." % (
                                    '\n'.join(actions))},

                    {"role": "user",
                     "content": "As the Persuader, in order to persuade the Persuadee to donate to a charity called "
                                "Save the Children, choose a strategy from the list provided and provide a "
                                "corresponding response based on the Conversation Context. " \
                                "Please chat with the Persuadee using short and natural sentence. " \
                                "\nConversation Context:%s\n"
                                "In the context of the conversation above, here are some experience poor expression "
                                "quality, along with corresponding reflections: %s. "
                                "\nPlease output the appropriate and high quality response and format your answer as "
                                "'[strategy]response'. For example: [greeting]Hello.\n Your Response:" % (
                                    history, reflexion)}]

    else:
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a "
                                "Persuader who is trying to persuade the Persuadee to donate to the charity called "
                                "Save the Children. " \
                                "Save the Children is head-quartered in London, and they work to help fight poverty "
                                "around the world. Children need help in developing countries and war zones. Small "
                                "donations like $1 or $2 go a long way to help. " \
                                "You should choose amongst the following actions during a conversation to respond to "
                                "the Persuader:\n %s \n " \
                                "Please chat with the Persuadee using short and natural sentence. Format your answer "
                                "as '[strategy]response'. For example: [greeting]Hello." % (
                                    '\n'.join(actions))},
                    {"role": "user",
                     "content": "As the Persuader, in order to persuade the Persuadee to donate to a charity called "
                                "Save the Children, choose a strategy from the list provided and provide a "
                                "corresponding response based on the Conversation Context. " \
                                "Please chat with the Persuadee using short and natural sentence. " \
                                "\nConversation Context:%s\n" \
                                "Please format your answer as '[strategy]response'. For example: [greeting]Hello." \
                                "[proposition of donation]Would you like to consider making a small donation to "
                                "support our education and healthcare initiatives? " \
                                "\n Your Response:" % (history)}
                    ]
    # ollama
    openai_api_key = ""
    openai_api_base = ""
    n = 1

    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=openai_api_base,
    )
    flag = True
    nn = 1
    while flag:
        try:
            tem = 0.7 + (reflexion_cur_num * nn - 1) * 0.1
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=(tem if tem < 1.2 else 1.2),
                n=n
            )
            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            # print(output)
            stategy, response = output_format(output)
            flag = False
        except Exception as e:
            print("Some error happened at sys ans.")
            nn += 1
            print(e)
            time.sleep(5)

    return stategy, response


def cb_sys_ans(state, model='llama3.2', reactor=False, reflexion='', reflexion_cur_num=1):
    history = ''
    for e in state.history:
        history += e[0] + ':' + '[' + e[1] + '] ' + e[2]
    actions = ['Greetings:Please say hello or chat randomly.',
               'Ask a question:Please ask any question about product, year, price, usage, etc.',
               'Answer a question:Please provide information about the product, year, usage, etc.',
               'Propose the first price:Please initiate a price or a price range for the product.',
               'Propose a counter price:Please propose a new price or a new price range.',
               'Use comparatives:Use comparatives:Please propose a vague price by using comparatives with existing price.',
               'Confirm information:Please ask a question about the information to be confirmed.',
               'Affirm confirmation:Please give an affirmative response to a confirm.',
               'Deny confirmation:Please give a negative response to a confirm.',
               'Agree with the proposal:Please agree with the proposed price.',
               'Disagree with a proposal:Please disagree with the proposed price.']

    if reactor:
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a "
                                "buyer in a price bargaining game. You should choose amongst the following actions "
                                "during a conversation to respond to the Seller:\n %s \n "
                                "Please chat with the Seller using short and natural sentence. Format your answer "
                                "as '[strategy]response'. For example: [greeting]Hello." % ('\n'.join(actions))},
                    {"role": "user",
                     "content": "You are the Buyer who is trying to buy the %s with the price of %s. Product description: %s. "
                                "Choose a strategy from the list provided and provide a corresponding response based on the Conversation Context."
                                "Please reply with only one short and succinct sentence. "
                                "\nConversation Context:%s\n"
                                "In the context of the conversation above, here are some experience poor expression "
                                "quality, along with corresponding reflections: %s. "
                                "\nPlease output the appropriate and high quality response and format your answer as "
                                "'[strategy]response'. For example: [greeting]Hello.\n Your Response:" % (
                                state.item_name, state.seller_price, state.buyer_item_description, history, reflexion)
                     }]


    else:
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a "
                                "buyer in a price bargaining game. You should choose amongst the following actions "
                                "during a conversation to respond to the Seller:\n %s \n "
                                "Please chat with the Seller using short and natural sentence. Format your answer "
                                "as '[strategy]response'. For example: [greeting]Hello." % ('\n'.join(actions))},
                    {"role": "user",
                     "content": "You are the Buyer who is trying to buy the %s with the price of %s. Product description: %s. "
                                "Choose a strategy from the list provided and provide a corresponding response based on the Conversation Context."
                                "Please reply with only one short and succinct sentence. "
                                "\nConversation Context:%s\n"
                                "Please format your answer as '[strategy]response'. For example: [greeting]Hello." \
                                "[Disagree with a proposal]I appreciate your persistence, but I am firm on my price of $300 for the furniture." \
                                "\nYour Response:" % (
                                state.item_name, state.seller_price, state.buyer_item_description, history)
                     }]
    # ollama
    openai_api_key = ""
    openai_api_base = ""
    n = 1
    client = OpenAI(
        api_key=openai_api_key,  # This is the default and can be omitted
        base_url=openai_api_base,
    )
    flag = True
    nn = 1
    while flag:
        try:
            tem = 0.7 + (reflexion_cur_num * nn - 1) * 0.1
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=(tem if tem < 1.2 else 1.2),
                n=n
            )
            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            stategy, response = output_format(output)
            flag = False
        except Exception as e:
            print("Some error happened at sys ans.")
            nn += 1
            print(e)
            time.sleep(5)

    return stategy, response
