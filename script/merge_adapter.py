
import peft
from transformers import AutoModel, AutoTokenizer


def merge_adapter_with_base_model(base_model_name: str, adapter_path: str, output_model_path: str):
    """
    Merges an adapter with a base model using the PEFT library and saves the merged model.

    :param base_model_name: The name or path of the base model.
    :param adapter_path: The path to the adapter.
    :param output_model_path: The path to save the merged model.
    """
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    peft_model = peft.AutoPeftModelForCausalLM.from_pretrained(adapter_path)
    print(type(peft_model))

    merged_model = peft_model.merge_and_unload() 
    
    # Save the merged model
    merged_model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"Merged model saved to {output_model_path}")


if __name__ == "__main__":
    from fire import Fire
    Fire(merge_adapter_with_base_model)