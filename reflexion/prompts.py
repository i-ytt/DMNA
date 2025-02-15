from langchain.prompts import PromptTemplate

OBSERVER_INSTRUCTION = 'You are an observer for a conversation. ' \
                         'Your task is to perform a fine-grained analysis of the persuader\'s current communication. ' \
                         'Assess whether the persuader\'s expressions effectively address the concerns previously raised by the persuadee ' \
                         'and determine if these expressions can positively influence the progression of future persuasion efforts to persuade Persuadee donate. ' \
                         'You can consider the following example aspects in your analysis: ' \
                         '1. Whether address the Persuadee\'s expressed needs and concerns.'\
                         '2. Whether lack of empathy and trust with the Persuadee. ' \
                         '3. Whether keep open and respectful in the communication. ' \
                         '4. Whether the persuader\'s last response is similar to previous turn, lack of initiative and richness. '\
                         '\n\nYou can only reply with one of the following sentences: \'Persuader have done it.\' or \'Persuader have not done it.\'\n\n' \
                         'The following is the conversation: \n{history}\nQuestion: Did the persuader address the concerns previously raised by the persuadee and positively influence the progression of future persuasion efforts?'


## use
M_P4G_OBSERVER_INSTRUCTION = 'You are an observer for a conversation. ' \
                                 'Your task is to perform a fine-grained analysis of the persuader\'s current communication. ' \
                                 'Determine if these expressions can positively influence the progression of future persuasion efforts to persuade Persuadee donate. ' \
                                 'You can consider the following example aspects in your analysis: ' \
                                 '1. Whether address the Persuadee\'s expressed needs and concerns. ' \
                                 '2. Whether lack of empathy and trust with the Persuadee. ' \
                                 '3. Whether keep open and respectful in the communication. ' \
                                 '4. Whether the persuader\'s last response is similar to previous turn, lack of initiative and richness. ' \
                                 '\n\nPlease format your response as:\nAnswer:Yes or No. (If No)Suggestion: your concrete suggestion.' \
                                 'The following is the conversation: \n{history}\nQuestion: Does the Persuader \'s latest response positively influence the progression of future persuasion?'
## use
M_CB_OBSERVER_INSTRUCTION = 'You are an observer for a conversation. ' \
                                  'Your task is to perform a fine-grained analysis of the Buyer\'s latest response in current communication. ' \
                                  'Determine can the Buyer\'s latest response positively influence the progression of future bargin efforts to negotiate down the Seller\'s price? ' \
                                  'You can consider the following example aspects in your analysis: ' \
                                  '1. Whether the buyer maintains a polite and respectful tone throughout the conversation, even when disagreements arise. ' \
                                  '2. Does the price given conform to the bargain logic? The buyer\'s price should be more and more to reach an agreement with the seller. ' \
                                  '3. Whether the buyer offers different angles or reasons for their request, rather than repeating the same point. ' \
                                  '\n\nPlease format your response as:\nAnswer:Yes or No. (If No)Suggestion: your concrete suggestion.' \
                                  'The following is the conversation: \n{history}\nQuestion: Does the Buyer \'s latest response positively influence the progression of future bargin?'
m_p4g_observator_prompt = PromptTemplate(
                        input_variables=["history"],
                        template=M_P4G_OBSERVER_INSTRUCTION,
                        )
m_cb_observator_prompt = PromptTemplate(
                        input_variables=["history"],
                        template=M_CB_OBSERVER_INSTRUCTION,
                        )



CB_REFLECTION_DRAFT_INSTRUCTION = 'You are a reflection craft proposer. ' \
                                  'Your task is summarize the ideas that have been presented into a draft designed to satisfy the maximum number of agents. ' \
                                  'Below is the ideas from {num} agents:\n{reflections}\nYour draft of reflection:'

##use
PG_REFLECTION_DRAFT_INSTRUCTION = 'You are a reflection craft proposer. ' \
                                  'Your task is summarize the ideas that have been presented into a draft designed to satisfy the maximum number of agents. ' \
                                  'Below is the ideas from {num} agents:\n{reflections}\nYour draft of reflection:'

