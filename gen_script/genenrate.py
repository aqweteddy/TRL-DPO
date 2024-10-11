from vllm import LLM
import os
from vllm.sampling_params import SamplingParams
from datasets import load_dataset, disable_caching, concatenate_datasets


def prepare_chosen(dct: dict):
    prompt, chosen = [], []
    messages = dct['messages'][0]
    
    if messages[0]['role'] == 'system':
        messages = messages[1:]
    
    for i in range(1, len(messages), 2):
        prompt.append(messages[:i])
        chosen.append(messages[i]['content'].strip())
    return {'prompt': prompt, 'chosen': chosen}

def apply_chat_template(dct: dict, tok):
    return {'_prompt': tok.apply_chat_template(dct['prompt'], add_generation_prompt=True, tokenize=False)}


def main(
    data_path: str,
    model_path: str = None,
    output_path: str = 'result/output.jsonl',
    data_size: int = None
):
    disable_caching()
    
    if os.path.isdir(data_path):
        jsonls = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jsonl')]
        dss = []
        for jsonl in jsonls:
            dss.append(load_dataset('json', data_files=jsonl)['train'])
            dss[-1] = dss[-1].remove_columns([c for c in dss[-1].column_names if c != 'messages'])
            dss[-1] = dss[-1].add_column('src', [jsonl] * len(dss[-1]))
        ds = concatenate_datasets(dss)
    else:
        ds = load_dataset('json', data_files=data_path)['train']
    
    if data_size:
        ds = ds.shuffle(42).select(range(data_size))
    
    ds = ds.map(prepare_chosen, num_proc=12, remove_columns=ds.column_names, batch_size=1, batched=True)
    llm = LLM(model=model_path, tokenizer_mode='auto', tensor_parallel_size=4)
    ds = ds.map(apply_chat_template, num_proc=12, fn_kwargs={'tok': llm.get_tokenizer()})
    
    # batched generation
    rejected = [] 
    for i in range(0, len(ds), 10000):
        print(f'Generating from {i} to {i+10000}')
        tmp = llm.generate(ds['_prompt'][i:i+10000], sampling_params=SamplingParams(max_tokens=16384, 
                                                                         top_p=0.5, 
                                                                         top_k=50, 
                                                                         temperature=0.7)
                        )
        rejected.extend([r.outputs[0].text for r in tmp])
    
    ds = ds.remove_columns([col for col in ds.column_names if col.startswith('_')])
    ds = ds.add_column('rejected', rejected)
    ds.to_json(output_path, force_ascii=False)


if __name__ == '__main__':
    from fire import Fire
    
    Fire(main)