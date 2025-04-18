from transformers import (
    AutoModelForCausalLM,  
    AutoProcessor,
    BitsAndBytesConfig
)
import torch
#code so my computer dose not burn up
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,  
)

import os
os.makedirs("offload_dir", exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision",
    device_map="auto", 
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    offload_folder="offload_dir",  
    offload_state_dict=True, 
)

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
#ask question test?:/
inputs = processor(text="hello what are you", return_tensors="pt")

inputs = {k: v.to(model.device) for k, v in inputs.items()}

inputs = {k: (v.to(dtype=torch.float16) if v.dtype == torch.uint8 else v) for k, v in inputs.items()}

with torch.no_grad():
    print("Input tensor properties:")
    for k, v in inputs.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
    
    out = model.generate(
        **inputs,
        max_new_tokens=100,
    )
print(processor.tokenizer.decode(out[0], skip_special_tokens=True))
