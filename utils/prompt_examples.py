from core.game import PersuasionGame, CBGame

EXP_DIALOG = [
	# extracted from 20180825-061105_792_live and 20180826-053845_531_live
	(PersuasionGame.SYS, PersuasionGame.S_Greeting,				"Hello. How are you?",),
	(PersuasionGame.USR, PersuasionGame.U_Neutral,				"I'm good, how are you doing?",),
	(PersuasionGame.SYS, PersuasionGame.S_TaskRelatedInquiry,	"Very well. I'm just up organizing info for my charity called Save the Children. Have you heard of this charity berfore?",),
	(PersuasionGame.USR, PersuasionGame.U_Neutral,				"No, I have not. Can you tell me more?",),
	(PersuasionGame.SYS, PersuasionGame.S_CredibilityAppeal,	"Save the Children is an organization that helps children in developing countries, by promoting children's rights and providing relief. It is an amazing charity that helps kids who are in desperate need. They can help with safety, education and more.",),
	(PersuasionGame.USR, PersuasionGame.U_NegativeReaction,		"That sounds great. I believe in this charity, but still wonder how much of the money I donate actually helps. I am always worried if I donate it will just go to some higer up that is living the high life.",),
	(PersuasionGame.SYS, PersuasionGame.S_EmotionAppeal,		"Every little bit makes a difference. When you have people who are so poor, it's amazing what a tiny amount can do. I usually donate in hopes I can at least feel like I did my part. If I donated and some corrupt person took it, that's the worst karma and even worst scandal imaginable.",),
	(PersuasionGame.USR, PersuasionGame.U_PositiveReaction,		"With that all said I do feel like any orginazation that aims to help the children I am more inclined to donate to them than most. I think helping children is an important thing as they are our future!",),
	(PersuasionGame.SYS, PersuasionGame.S_PropositionOfDonation,"I think donating to this cause would def be a step in the right direction to hopefully helping across the world the children that are in dispair. I don't want you to donate any more than you want, so if you want to donate how much do you to do?",),
	(PersuasionGame.USR, PersuasionGame.U_Donate,				"I would donate 1 dollar to this charity and feel good about it I think.",),
]

CB_EXP_DIALOG = [
	# extracted from 479th dialog in ESConv
 	# (EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello.",),
	# (CBGame.SYS, CBGame.S_Greetings, 					"Hello.",),
	(CBGame.SYS, CBGame.S_Greetings,					"Hi, man thats a nice trak?",),
	(CBGame.USR, CBGame.U_PositiveReaction,				"Yes, its really nice.",),
	(CBGame.SYS, CBGame.S_ProposeTheFirstPrice,			"How about you give it to me for 8000?",),
	(CBGame.USR, CBGame.U_NegativeReaction,				"That is way too little.",),
	(CBGame.SYS, CBGame.S_ProposeACounterPrice,			"I realy dont need it but I can pay you 8200 if you let it go.",),
	(CBGame.USR, CBGame.U_Neutral,						"I can go down to 14k but I can't knock off 6k.",),
	(CBGame.SYS, CBGame.S_ProposeACounterPrice,			"Oh man i cant afford that, I need to buy food for my pigs and farm and send my child to private school, can you do 13k?",),
	(CBGame.USR, CBGame.U_Deal,							"I will take 13k.",),
	(CBGame.SYS, CBGame.S_ConfirmInformation,			"Ok, I can pick it up this weekend.",),
	(CBGame.USR, CBGame.U_Deal,							"Sounds good.",),
]