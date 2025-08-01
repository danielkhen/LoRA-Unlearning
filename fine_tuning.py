import re

from peft import LoraConfig, get_peft_model
from torch.nn import Linear, Embedding, Conv2d, Conv1d

LORA_LAYERS = [Linear, Embedding, Conv2d, Conv1d]

def get_lora_model(model, rank, target_regexes):
    # Find all modules whose name matches any regex in target_regexes and whose type is in LORA_LAYERS
    target_modules = [
        name for name, module in model.named_modules()
        if any(re.fullmatch(pattern, name) for pattern in target_regexes) and type(module) in LORA_LAYERS
    ]
    
    # Complementary list: modules not in target_modules
    modules_to_save = [
        name for name, module in model.named_modules()
        if name not in target_modules and type(module) in LORA_LAYERS
    ]

    lora_config = LoraConfig(r=rank, target_modules=target_modules, modules_to_save=modules_to_save)
    model = get_peft_model(model, lora_config)
    
    return model