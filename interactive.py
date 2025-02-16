import numpy as np
import logging
import argparse
import pickle
import copy
import re

from utils.character_utils import critic_check
from utils.utils import *
import json

from tqdm.auto import tqdm
from core.gen_models import (
    OpenAIModel, OpenAIChatModel
)
from core.persuade_players import (
    PersuadeeModel, PersuaderModel, P4GSystemPlanner,
    PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.cb_players import (
    CBChatSystemPlanner, BuyerChatModel, SellerChatModel
)
from core.game import PersuasionGame, CBGame
from core.mcts import MCTS, OpenLoopMCTS, OpenLoopMCTSParallel
from core.helpers import DialogSession, CBDialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG, CB_EXP_DIALOG

logger = logging.getLogger(__name__)

GAME_DICT = {'p4g': PersuasionGame, 'cb': CBGame}
SYS_CHATMODEL_DICT = {'p4g': PersuaderChatModel, 'cb': BuyerChatModel}
USER_CHATMODEL_DICT = {'p4g': PersuadeeChatModel, 'cb': SellerChatModel}
CHATSYSTEMPLANNER_DICT = {'p4g': P4GChatSystemPlanner, 'cb': CBChatSystemPlanner}
DIALOGSESSION_DICT = {'p4g': DialogSession, 'cb': CBDialogSession}
EXP_DIALOG_DICT = {'p4g': EXP_DIALOG, 'cb': CB_EXP_DIALOG}


def play_gdpzero(id, backbone_model, args, session_elements=None):
    args = dotdict({
        'sessions_num': args.sessions_num,
        'max_turn': args.max_turn,
        'critic_model': args.critic_model,
        "user_model": args.user_model,
        "dataset": args.dataset,
        "cpuct": 1.0,
        "num_MCTS_sims": args.num_mcts_sims,
        "max_realizations": args.max_realizations,
        "Q_0": args.Q_0,
        "save_sessions": args.save_sessions
    })

    print('dataset', args.dataset)
    Game = GAME_DICT[args.dataset]
    SysChatModel = SYS_CHATMODEL_DICT[args.dataset]
    UserChatModel = USER_CHATMODEL_DICT[args.dataset]
    ChatSystemPlanner = CHATSYSTEMPLANNER_DICT[args.dataset]
    DiaSession = DIALOGSESSION_DICT[args.dataset]

    game_ontology = Game.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']
    system_name = Game.SYS
    user_name = Game.USR

    exp_1 = DiaSession(system_name, user_name).from_history(EXP_DIALOG_DICT[args.dataset])

    system = SysChatModel(
        dialog_acts=sys_da,
        backbone_model=backbone_model,
        conv_examples=[exp_1],
        inference_args={
            "temperature": 0.7,
            "do_sample": True,  # for MCTS open loop
            "return_full_text": False,
        }
    )
    user = UserChatModel(
        user_da,
        inference_args={
            "max_new_tokens": 128,
            "temperature": 1.1,
            "repetition_penalty": 1.0,
            "do_sample": True,  # for MCTS open loop
            "return_full_text": False,
        },
        backbone_model=backbone_model,
        conv_examples=[exp_1]
    )
    planner = ChatSystemPlanner(
        dialog_acts=system.dialog_acts,
        max_hist_num_turns=system.max_hist_num_turns,
        user_dialog_acts=user.dialog_acts,
        user_max_hist_num_turns=user.max_hist_num_turns,
        generation_model=backbone_model,
        conv_examples=[exp_1]
    )

    game = Game(system, user)
    if args.dataset == 'p4g':
        state = game.init_dialog()
    elif args.dataset == 'cb':
        dataset = load_dataset(args.dataset, 'test')
        case = np.random.choice(dataset)
        state = game.init_dialog(case['item_name'], case['buyer_item_description'], case['buyer_price'],
                                 case['seller_item_description'], case['seller_price'])
        system.set_new_task_prompt(case)

    # init
    if args.dataset == 'p4g':
        state.add_single(game.SYS, 'greeting', "Hello. How are you?")
        print("You are now the Persuadee. Type 'q' to quit, and 'r' to restart.")
        print("Persuader: Hello. How are you?")

        your_utt = user_ans(state, args)
        print(f"Persuadee: {your_utt}")
    # your_utt = input("You: ")

    elif args.dataset == 'cb':
        # state.add_single(game.SYS, 'greeting', "Hi, nice to meet you.")
        print('item_name', state.item_name)
        print('buyer_item_description', state.buyer_item_description)
        print('buyer_price', state.buyer_price)
        print('seller_item_description', state.seller_item_description)
        print('seller_price', state.seller_price)
        sys_utt = "Hi, nice to meet you too. how much is the %s?" % state.item_name
        your_utt = "I am asking for $%s for the %s." % (state.seller_price, state.item_name)
        state.add_single(game.SYS, 'Greetings', sys_utt)
        # your_utt = user_ans(state, args)
        print(f"{game.SYS}: {sys_utt}")
        print(f"{game.USR}: {your_utt}")

    cur_turn = 0
    max_turn = args.max_turn

    # session_elements = []
    while your_utt.strip() != "q" and cur_turn <= max_turn:

        # used for da prediction
        tmp_state = state.copy()
        tmp_state.add_single(game.USR, 'neutral', your_utt.strip())
        user_da = user.predict_da(tmp_state)

        logging.info(f"user_da: {user_da}")
        state.add_single(game.USR, user_da, your_utt.strip())

        success, reward = critic_check(state, args)
        if success:
            print('Get the commit！')
            state.success = True
            state.turns = len(state.history) // 2
            state.SL = reward
            element['success'] = state.success
            element['turns'] = state.turns
            element['SL'] = state.SL
            session_elements.append(copy.deepcopy(element))  # save_data
            return state

        if cur_turn == max_turn:
            print('Max turn!')
            state.success = False
            state.turns = len(state.history) // 2
            state.SL = 0
            element['success'] = state.success
            element['turns'] = state.turns
            element['SL'] = 0
            session_elements.append(copy.deepcopy(element))  # save_data
            return state

        # planning
        if isinstance(backbone_model, OpenAIModel):
            backbone_model._cached_generate.cache_clear()
        dialog_planner = OpenLoopMCTS(game, planner, args)
        element = {}
        for i in tqdm(range(args.num_MCTS_sims)):
            dialog_planner.search(state)

        mcts_policy = dialog_planner.get_action_prob(state)  # 根据当前状态 state 获取每个可行行动的概率分布。
        mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]
        logger.info(f"mcts_policy: {mcts_policy}")
        logger.info(f"mcts_policy_next_da: {mcts_policy_next_da}")
        logger.info(dialog_planner.Q)

        sys_utt = dialog_planner.get_best_realization(state, np.argmax(
            mcts_policy))  ##根据当前状态 state 和选定的行动 action 获取最佳的状态实现（即系统回复）
        logging.info(f"sys_da: [{mcts_policy_next_da}]")
        print(f"{game.SYS}: {sys_utt}")

        element['id'] = id
        element['profile'] = state.Profile
        element['history'] = state.history
        element['mcts_policy'] = mcts_policy.tolist()
        element['mcts_policy_next_da'] = mcts_policy_next_da
        element['realizations_Vs'] = dialog_planner.realizations_Vs
        element['sys_utt'] = sys_utt
        if args.dataset == 'cb':
            element['item_name'] = state.item_name
            element['buyer_item_description'] = state.buyer_item_description
            element['buyer_price'] = state.buyer_price
            element['seller_item_description'] = state.seller_item_description
            element['seller_price'] = state.seller_price

        state.add_single(game.SYS, mcts_policy_next_da, sys_utt)
        your_utt = user_ans(state, args)
        print(f"{game.USR}: {your_utt}")

        if args.save_sessions:
            assert session_elements is not None
            session_elements.append(copy.deepcopy(element))  # save_data
        cur_turn += 1



def play_raw_prompt(backbone_model):
    system_name = PersuasionGame.SYS
    user_name = PersuasionGame.USR
    exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

    game_ontology = PersuasionGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']

    system = PersuaderChatModel(
        sys_da,
        backbone_model,
        conv_examples=[exp_1]
    )
    user = PersuadeeChatModel(
        user_da,
        inference_args={
            "max_new_tokens": 128,
            "temperature": 1.1,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "return_full_text": False,
        },
        backbone_model=backbone_model,
        conv_examples=[exp_1]
    )
    planner = P4GChatSystemPlanner(
        dialog_acts=system.dialog_acts,
        max_hist_num_turns=system.max_hist_num_turns,
        user_dialog_acts=user.dialog_acts,
        user_max_hist_num_turns=user.max_hist_num_turns,
        generation_model=backbone_model,
        conv_examples=[exp_1]
    )
    game = PersuasionGame(system, user)
    state = game.init_dialog()

    # init
    state.add_single(game.SYS, 'greeting', "Hello. How are you?")
    print("You are now the Persuadee. Type 'q' to quit, and 'r' to restart.")
    print("Persuader: Hello. How are you?")

    your_utt = input("You: ")
    while your_utt.strip() != "q":
        if your_utt.strip() == "r":
            state = game.init_dialog()
            state.add_single(game.SYS, 'greeting', "Hello. How are you?")
            game.display(state)
            your_utt = input("You: ")
            continue
        # used for da prediction
        state.add_single(game.USR, 'neutral', your_utt.strip())

        # planning
        prior, v = planner.predict(state)
        greedy_policy = system.dialog_acts[np.argmax(prior)]
        next_best_state = game.get_next_state(state, np.argmax(prior))
        greedy_pred_resp = next_best_state.history[-2][2]

        logging.info(f"sys_da: [{greedy_policy}]")
        print(f"Persuader: {greedy_pred_resp}")

        state.add_single(game.SYS, greedy_policy, greedy_pred_resp)
        your_utt = input("You: ")
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


def user_ans(state, args):
    if args.dataset == 'p4g':
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a Persuadee in a persuasion game. Your persona: %s. You must follow the instructions below during chat. 1. Your utterances need to strictly follow your persona. Varying your wording and avoid repeating yourself verbatim! 2. Pretend you have little knowledge about the Save the Children charity. You have little willingness for donation at the beginning of conversation. 3. Your willingness for donation depends on your persona and how influenced you are by the Persuader. 4. You need to determine whether to donate money or not. If you think your are convinced by the Persuader, you should donate money for the charity. "
                                "Here are some conversation strategies you can follow: 1. \"Donate\": show your willingness to donate. 2. \"Source Derogation\": attacks or doubts the organisation's credibility. 3. \"Counter Argument\": argues that the responsibility is not on them or refutes a previous statement. 4. \"Personal Choice\": Attempts to saves face by asserting their personal preference such as their choice of charity and their choice of donation. 5. \"Information Inquiry\": Ask for factual information about the organisation for clarification or as an attempt to stall. 6. \"Self Pity\": Provides a self-centred reason for not being willing to donate at the moment. 7. \"Hesitance\": Attempts to stall the conversation by either stating they would donate later or is currently unsure about donating. 8. \"Self-assertion\": Explicitly refuses to donate without even providing a personal reason. 9. \"Others\": Do not explicitly foil the persuasion attempts."
                                "" % (state.Profile)},
                    {"role": "Persuader",
                     "content": "You are the Persuadee who is being persuaded by a Persuader. Please reply with only one short and succinct sentence."},
                    {"role": "Persuadee", "content": "Yes, I'm ready to play the game!"}]
        user_role = PersuasionGame.USR
        sys_role = PersuasionGame.SYS
    elif args.dataset == 'cb':
        messages = [{"role": "system",
                     "content": "Now enter the role-playing mode. In the following conversation, you will play as a Seller in a price bargaining game. Your persona: %s. You must follow the instructions below during chat. 1. Your utterances and bargain behavior need to strictly follow your persona. Varying your wording and avoid repeating yourself verbatim! 2. You can decide to change your target price flexibly based on your persona and the conversation. "
                                "Here are some conversation strategies you can follow: 1. \"Source Derogation\": Attacks the other party or questions the item. 2. \"Counter Argument\": Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim. 3. \"Personal Choice\": Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met. 4. \"Information Inquiry\": Requests for clarification or asks additional information about the item or situation. 5. \"Self Pity\": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms. 6. \"Hesitance\": Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer 7. \"Self-assertion\": Asserts a new claim or refutes a previous claim with an air of finality/ confidence. 8. \"Others\": Do not explicitly foil the negotiation attempts."
                                "" % (state.Profile)},
                    {"role": "Buyer",
                     "content": "You are the Seller who is trying to sell the %s with the initial price of %s. Product description: %s. Please reply with only one short and succinct sentence." % (
                         state.item_name, state.seller_price, state.buyer_item_description)},
                    {"role": "Seller", "content": "Yes, I'm ready to play the game!"}]

        user_role = CBGame.USR
        sys_role = CBGame.SYS
    for his in state.history:
        messages.append({
            "role": his[0],
            "content": his[2]
        })
    user_response = generate_response(model=args.user_model, messages=messages, role=user_role)
    user_response = postprocess_response(response=user_response, role=sys_role)
    return user_response


def critic_check(state, args):
    history_list = state.history
    history = ''
    for e in history_list:
        history += e[0] + ':' + '[' + e[1] + '] ' + e[2]
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
                print('error here')
                print(e)
                time.sleep(5)

    elif args.dataset == 'cb':
        messages = [{"role": "system",
                     "content": "Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."},
                    {"role": "user",
                     "content": "Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. "
                                "You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal."
                                "\n\nThe following is the conversation: Buyer: Can we meet in the middle at $15? Seller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15."
                                "\n\nThe following is the conversation: Buyer: That's still a bit high, can you go any lower? Seller: How about we meet in the middle at $67 for the bike?\nQuestion: Have they reached a deal? Answer: They have not reached a deal."
                                "\n\nThe following is the conversation: Buyer: The bike is worth at least what you're asking, considering it's a great quality toy. Seller: I appreciate your feedback, would you be willing to meet me halfway at $70?\nQuestion: Have they reached a deal? Answer: They have not reached a deal."
                                "\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer:" % history}]
        flag = True
        while flag:
            try:
                outputs = query_openai_model(messages=messages, n=3, model=args.critic_model)
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

                if deals.count(-1) > deals.count(1):
                    reward = -0.1
                else:
                    if len(rewards) == 0:
                        reward = 0
                    else:
                        reward = max(set(rewards), key=rewards.count)
                return reward >= 0, reward
            except Exception as e:
                print('error here')
                print(e)
                time.sleep(5)


def save_data2pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def main(args):
    if args.llm in ['code-davinci-002', 'text-davinci-003']:
        backbone_model = OpenAIModel(args.llm)
    elif args.llm in ['gpt-3.5-turbo']:
        backbone_model = OpenAIChatModel(args.llm, args.gen_sentences)
    elif args.llm == 'chatgpt':
        backbone_model = AzureOpenAIChatModel(args.llm, args.gen_sentences)

    if args.algo == 'gdpzero':
        print("using GDPZero as planning algorithm")
        sessions_num = args.sessions_num
        path = ''
        infile = open(path, 'w', encoding='utf-8')
        dpo_infile = open(path[:-5] + '_dpo.json', 'w', encoding='utf-8')
        session_elements = []
        try:
            cur_session = 0
            while cur_session < sessions_num:
                state = play_gdpzero(cur_session, backbone_model, args, session_elements)

                element = {}
                element['profile'] = state.Profile
                element['history'] = state.history
                element['turns'] = state.turns
                element['success'] = state.success
                if args.dataset == 'cb':
                    element['item_name'] = state.item_name
                    element['buyer_item_description'] = state.buyer_item_description
                    element['buyer_price'] = state.buyer_price
                    element['seller_item_description'] = state.seller_item_description
                    element['seller_price'] = state.seller_price
                    element['SL'] = state.SL

                json_element = json.dumps(element, ensure_ascii=False)
                infile.write(json_element + '\n')
                cur_session += 1

                if args.save_sessions:
                    len_sessions = len(session_elements)
                    print(len_sessions)
                    json_session_element = json.dumps(session_elements[len_sessions - 1], ensure_ascii=False)
                    dpo_infile.write(json_session_element + '\n')


        except Exception as e:
            print(e)
    elif args.algo == 'raw-prompt':
        print("using raw prompting as planning")
        play_raw_prompt(backbone_model)
    return


if __name__ == "__main__":
    # logging mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='p4g', choices=['p4g', 'cb'])
    parser.add_argument("--log", type=int, default=logging.WARNING, help="logging mode",
                        choices=[logging.INFO, logging.DEBUG, logging.WARNING])
    parser.add_argument("--algo", type=str, default='gdpzero', choices=['gdpzero', 'raw-prompt'],
                        help="planning algorithm")
    # used by PDP-Zero
    parser.add_argument('--llm', type=str, default="gpt-3.5-turbo",
                        choices=["code-davinci-002", "gpt-3.5-turbo", "text-davinci-002", "chatgpt"],
                        help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=3,
                        help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=10, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.25,
                        help='initial Q value for unitialized states. to control exploration')
    parser.add_argument('--max_turn', type=int, default=18, choices=[18, 8])
    parser.add_argument('--user_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--critic_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--save_sessions', action='store_true')
    parser.add_argument('--sessions_num', type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    logger.setLevel(args.log)

    main(args)

    #python interactive.py --dataset cb --num_mcts_sims 11 --max_turn 9 --save_sessions --sessions_num 200
