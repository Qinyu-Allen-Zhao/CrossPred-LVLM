def build_model(args):
    if args.model_name == "InstructBLIP":
        from .InstructBLIP import InstructBLIP
        model = InstructBLIP(args)
    elif args.model_name == "LLaVA":
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "LLaMA_Adapter":
        from .LLaMA_Adapter import LLaMA_Adapter
        model = LLaMA_Adapter(args)
    elif args.model_name == "MMGPT":
        from .MMGPT import MMGPT
        model = MMGPT(args)
    elif args.model_name == "GPT4":
        from .GPT4 import GPTClient
        model = GPTClient(args)
    elif args.model_name == "Gemini":
        from .Gemini import GeminiClient
        model = GeminiClient(args)
    elif args.model_name == "MiniGPT4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif args.model_name == "mPLUG-Owl":
        from .mPLUG_Owl import mPLUG_Owl
        model = mPLUG_Owl(args)
    elif args.model_name == "Cobra":
        from .Cobra import Cobra
        model = Cobra()
    elif args.model_name == "Qwen-VL":
        from .Qwen_VL import Qwen_VL
        model = Qwen_VL(args)
    elif args.model_name == "Cambrian":
        from .Cambrian import Cambrian
        model = Cambrian(args)
    elif args.model_name == "Fuyu":
        from .fuyu import Fuyu
        model = Fuyu(args)
    elif args.model_name == "Kosmos2":
        from .kosmos2 import Kosmos2
        model = Kosmos2(args)
    elif args.model_name == "OpenFlamingo":
        from .openflamingo import OpenFlamingo
        model = OpenFlamingo(args)
    elif args.model_name == "BLIP2":
        from .BLIP2 import BLIP2
        model = BLIP2(args)
    elif args.model_name == "InternLM_XComposer":
        from .InternLM_XComposer import InternLM_XComposer
        model = InternLM_XComposer(args)
    elif args.model_name == "prismatic":
        from .Prismatic import PrismaticVLM
        model = PrismaticVLM(args)
    elif args.model_name == "CLIP":
        from .CLIP import CLIP
        model = CLIP(args)
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    return model
