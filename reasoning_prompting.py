from OpenAIInterface_new import OpenAIInterface as oAI
from omegaconf import OmegaConf
import json

def main():
    data_address = "eedi_reasoning_prompts.json"
    with open (data_address) as f:
        data = json.load(f)
    
    prompts = data["prompts"]
    gt_difficulties = data["difficulties"]
    
    # print max and min difficulty
    print("Max difficulty:", max(gt_difficulties))
    print("Min difficulty:", min(gt_difficulties))
    
    # prompts = prompts[40:70]
    
    oaicfg = {
        'use_azure': False,
        'model': "gpt-4o", #   gpt-4o, gpt-4-turbo o1-preview
        'temperature' : 0.7,
        'max_tokens' : 1000, 
        'top_p' : 0.9,
        'frequency_penalty' : 0.0,
        'presence_penalty' : 0.0,
        'stop' : [],
        'logprobs': None,
        'echo' : False
        }
    conf = OmegaConf.create(oaicfg)        
    
    predictions = oAI.getCompletionForAllPrompts(conf, prompts, batch_size=20, use_parallel=True)
    oAI.save_cache()
    
    result = {"predictions": predictions, "gt_difficulties": gt_difficulties, "prompts": prompts}
    result_address = f"eedi_generated_reasonings_4o.json"
    
    
    with open(result_address, 'w') as f:
        json.dump(result, f, indent=4)
    
    

if __name__ == "__main__":
    main()
