import random
import re

import numpy as np
import torch.cuda
import fire

from promptcache.model import Llama2, Falcon, Mpt, CodeLlama

from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template


def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)


def main(enable_cache=True):
    enable_cpu_inference = False
    disable_prompt_cache = not enable_cache

    lm_for_cache = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                             load_in_8bit=True,
                             device_map="auto")

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                       load_in_8bit=False,
                       device_map=None)

    # lm = Falcon("tiiuae/falcon-7b-instruct",
    #             load_in_8bit=True if not disable_cuda else False,
    #             device_map="auto" if not disable_cuda else None)

    # lm = Mpt("mosaicml/mpt-7b-chat-8k",
    #          load_in_8bit=True if not disable_cuda else False,
    #          device_map="auto" if not disable_cuda else None)

    cache_engine = CacheEngine(5000, lm_for_cache, target_device='cpu' if enable_cpu_inference else None)
    gen_engine = GenerationEngine(lm)

    preproc = [
        # CompactSpaces(),
        lm.get_formatter()
    ]

    # cache_engine.add_schema(read_file("./examples/poison.xml", preproc), max_tokens=800)
    cache_engine.add_schema(read_file("./examples/code_generation_game.xml", preproc), max_tokens=800)

    parameter = GenerationParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    # prompt_text = f"""
    #     <prompt schema='code-generation-game'>
    #     <unit.py/>
    #     <map.py/>
    #     <player.py/>
    #     <game.py/>
    #     <database.py/>
    #     <user>
    #         Explain the game logic briefly.
    #     </user>
    #     </prompt>
    #     """
        
    prompt_text = f"""
        <prompt schema='code-generation-game'>
        <user>
            Hi. 
        </user>
        </prompt>
        """

    prompt = Prompt(prompt_text, preproc)
    # Taojie: 这一步已经生成了KV cache，based on the prompt, this function is critical!
    # no_cache = False
    # Question: 是不是包括了system的prompt呢？ 按理来说是的，否则应该不会生成它的文本。
    token_ids, position_ids, cache_time, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache,
                                                                      return_full_position_ids=lm.use_full_position_ids)
    
    print("\ndebug:")
    # The following tokens are the user token: "Create a main entry for the game:"
    print(f"token_ids: {token_ids}") # [6204, 263, 1667, 6251, 363, 278, 3748, 29901, 13, 4706, 518, 29914, 25580, 29962]
    print(f"position_ids: {position_ids}") # [4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347]
    # print(f"cache_time: {cache_time}")  # 316.2 ms. TODO: why differnt from prefill latency?
    # print(f"type of cache: {type(cache)}")  # list 
    # print(f"len of cache: {len(cache)}")  # 32: number of layers
    layer_0 = cache[0]
    layer0_keys = layer_0[0]
    # print(f"len of layer_0: {len(layer_0)}")  # 2
    # print(f"type of layer_0: {type(layer_0)}")  # tuple
    print(f"type of layer0_keys: {type(layer0_keys)}")  # torch.Tensor
    print(f"shape of layer0_keys: {layer0_keys.shape}")  # torch.Size([32, 922, 128])
    

    output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2,
                                        use_full_position_ids=lm.use_full_position_ids)

    print(f"Assistant: ", end="", flush=True)

    resp = ""
    pre = 0
    for outputs in output_stream:
        output_text = outputs.new_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            tt = " ".join(output_text[pre:now])
            resp += tt + " "
            print(tt, end=" ", flush=True)
            pre = now
    tt = " ".join(output_text[pre:])
    print(tt, flush=True)
    resp += tt

    print("\n")
    prompt_text += f"<assistant>{resp}</assistant>"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    fire.Fire(main)
