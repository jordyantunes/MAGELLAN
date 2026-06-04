import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained('/LLMs/flan-t5-base', quantization_config=config, device_map='auto', local_files_only=True)
print('Model device map:', model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A')
for name, param in list(model.named_parameters())[:5]:
    print(f'{name}: {param.device}')
print('VRAM used (MB):', torch.cuda.memory_allocated() / 1e6)
