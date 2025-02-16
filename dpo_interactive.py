import os

import numpy as np

from core.cb_players import BuyerChatModel, CBChatSystemPlanner, SellerChatModel
from utils.character_utils import user_ans, critic_check, pg_sys_ans, cb_sys_ans
from utils.utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from reflexion.ExpReflexion import ExpReflexion
import argparse

import json

from core.gen_models import (
    OpenAIModel, OpenAIChatModel
)
from core.persuade_players import (
    PersuadeeModel, PersuaderModel, P4GSystemPlanner,
    PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame, CBGame
from core.helpers import DialogSession, CBDialogSession
from utils.prompt_examples import EXP_DIALOG, CB_EXP_DIALOG

GAME_DICT = {'p4g': PersuasionGame, 'cb': CBGame}
SYS_CHATMODEL_DICT = {'p4g': PersuaderChatModel, 'cb': BuyerChatModel}
USER_CHATMODEL_DICT = {'p4g': PersuadeeChatModel, 'cb': SellerChatModel}
CHATSYSTEMPLANNER_DICT = {'p4g': P4GChatSystemPlanner, 'cb': CBChatSystemPlanner}
DIALOGSESSION_DICT = {'p4g': DialogSession, 'cb': CBDialogSession}
EXP_DIALOG_DICT = {'p4g': EXP_DIALOG, 'cb': CB_EXP_DIALOG}
REF_BEGIN_TURN_DICT = {'p4g': 0, 'cb': 0}


def play_gdpzero_dpo(backbone_model, args, logger):
    session_experience = []
    reflexion_num_sum = 0

    logger.write(f'dataset: {args.dataset}')
    Game = GAME_DICT[args.dataset]
    SysChatModel = SYS_CHATMODEL_DICT[args.dataset]
    UserChatModel = USER_CHATMODEL_DICT[args.dataset]

    game_ontology = Game.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']
    user_da = game_ontology['user']['dialog_acts']
    system_name = Game.SYS
    user_name = Game.USR

    exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG_DICT[args.dataset])

    system = SysChatModel(
        sys_da,
        backbone_model,
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
        your_utt = user_ans(state, args)

        logger.write("Persuader: Hello. How are you?")
        logger.write(f"Persuadee: {your_utt}")
    elif args.dataset == 'cb':
        sys_utt = "Hi, nice to meet you too. how much is the %s?" % state.item_name
        your_utt = "I am asking for $%s for the %s." % (state.seller_price, state.item_name)
        state.add_single(game.SYS, 'Greetings', sys_utt)
        logger.write(f'item_name: {state.item_name}')
        logger.write(f'buyer_item_description: {state.buyer_item_description}')
        logger.write(f'buyer_price: {state.buyer_price}')
        logger.write(f'seller_item_description: {state.seller_item_description}')
        logger.write(f'seller_price: {state.seller_price}')
        logger.write(f"{game.SYS}: {sys_utt}")
        logger.write(f"{game.USR}: {your_utt}")

    cur_turn = 0
    max_turn = args.max_turn

    exp = ExpReflexion(state, args.dataset)
    session_reflection_history = []
    while your_utt.strip() != "q" and cur_turn <= max_turn:

        # used for da prediction
        tmp_state = state.copy()
        tmp_state.add_single(game.USR, 'neutral', your_utt.strip())
        user_da = user.predict_da(tmp_state)

        # logging.info(f"user_da: {user_da}")
        state.add_single(game.USR, user_da, your_utt.strip())

        if cur_turn > 0:
            success, reward = critic_check(state, args)
            if success:
                logger.write('Get the commit！')
                state.success = True
                state.turns = len(state.history) // 2
                state.SL = reward
                state.ref_num_sum = reflexion_num_sum
                state.reflection_history = session_reflection_history
                return state, session_experience

            if cur_turn == max_turn:
                logger.write('Max turn!')
                state.success = False
                state.turns = len(state.history) // 2
                state.SL = 0
                state.ref_num_sum = reflexion_num_sum
                state.reflection_history = session_reflection_history
                return state, session_experience

        # planning
        if isinstance(backbone_model, OpenAIModel):
            backbone_model._cached_generate.cache_clear()

        if args.dataset == 'p4g':
            stategy, sys_utt = pg_sys_ans(state, args.model_name)
        elif args.dataset == 'cb':
            stategy, sys_utt = cb_sys_ans(state, args.model_name)

        state.add_single(game.SYS, stategy, sys_utt)
        logger.write(f"{state.SYS}: ({stategy}){sys_utt}")

        # expression reflexion
        ref_begin_turn = REF_BEGIN_TURN_DICT[args.dataset]
        if cur_turn > ref_begin_turn and args.reflexion:
            turn_reflection, reflexion_cur_num = exp.muti_reflexion(logger, dataset=args.dataset, max_reflexion=args.max_reflexion, single_critic=args.single_critic)
            session_reflection_history.append({'turn_id': cur_turn, 'turn_reflection': turn_reflection})
            exp.reset4memory()
            reflexion_num_sum += reflexion_cur_num


        your_utt = user_ans(state, args)
        logger.write(f"{state.USR}: {your_utt}")

        cur_turn += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cb', choices=['p4g', 'cb'])
    parser.add_argument("--log_dir", type=str, default='logs')
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
    parser.add_argument('--sessions_num', type=int, default=50, help='test session num')
    parser.add_argument('--user_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--critic_model', type=str, default='gpt-3.5-turbo-1106')
    parser.add_argument('--model_name', type=str, default='llama3.2', help='')
    parser.add_argument('--max_turn', type=int, default=12, choices=[18, 9, 12])
    parser.add_argument('--save_exp', action='store_true', help='save experience or not.')
    parser.add_argument('--reflexion', action='store_true', help='use reflexion')
    parser.add_argument('--single_critic', action='store_true', help='whether single critic')
    parser.add_argument('--max_reflexion', type=int, default=7)
    args = parser.parse_args()


    backbone_model = OpenAIChatModel('gpt-3.5-turbo')

    try:
        path = ''
        logger = Logger(path[:-5] + ".txt")
        logger.write(path)
        session_infile = open(path, 'w', encoding='utf-8')
        cur_session = 0
        session_experiences = []
        while cur_session < args.sessions_num:
            logger.write(f'No.{cur_session}---------------------------------------------------------------------')
            state, session_experience = play_gdpzero_dpo(backbone_model, args, logger)
            logger.write(f'-------------------------------------------------------------------------------------')

            element = {}
            element['id'] = cur_session
            element['profile'] = state.Profile
            element['history'] = state.history
            element['turns'] = state.turns
            element['success'] = state.success
            element['SL'] = state.SL
            element['ref_num_sum'] = state.ref_num_sum
            element['reflection_history'] = state.reflection_history
            json_element = json.dumps(element, ensure_ascii=False)
            session_infile.write(json_element + '\n')
            if args.save_exp and len(session_experience) != 0:
                session_experiences.extend(session_experience)

            cur_session += 1


    except Exception as e:
        print(e)
