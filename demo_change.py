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

    cache_engine = CacheEngine(5000, lm_for_cache, target_device='cpu' if enable_cpu_inference else None)
    gen_engine = GenerationEngine(lm)

    preproc = [
        lm.get_formatter()
    ]

    cache_engine.add_schema(read_file("./examples/code_generation_game.xml", preproc), max_tokens=800)


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
