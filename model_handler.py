from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision",
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    offload_folder="offload_dir",
    offload_state_dict=True,
)

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

def respond(prompt_text):
    inputs = processor(text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs = {k: (v.to(dtype=torch.float16) if v.dtype == torch.uint8 else v) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return processor.tokenizer.decode(output[0], skip_special_tokens=True)
