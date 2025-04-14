import json
import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModel, AutoTokenizer
from transformers import LongformerModel, LongformerTokenizer
import argparse
from utils import initialize_seeds, get_data_for_LLM, make_prompt, count_greater_pairs, calculate_r2, draw, seed_worker
import numpy as np
from tqdm import tqdm
import copy
import math
import wandb


device = torch.device('cuda:0')
initialize_seeds(45)
 
class StandardDataset(Dataset):
    def __init__(self, questions):
        self.data = [
            {
                "prompt": make_prompt(question),
                # "label": question["difficulty"]
                "label": question["GroundTruthDifficulty"],
                "ID": question["ID"]
            }
            for question in questions
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
    
class StandardCollator:
    # def __init__(self, tokenizer: AutoTokenizer):
    def __init__(self, tokenizer: LongformerTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        all_prompts = [sample["prompt"] for sample in batch]
        all_labels = [sample["label"] for sample in batch]
        all_ids = [sample["ID"] for sample in batch]
        # prompts_tokenized = self.tokenizer(all_prompts, padding='longest', return_tensors='pt', max_length = 512, truncation = True).to(device)
        prompts_tokenized = self.tokenizer(all_prompts, padding='longest', return_tensors='pt', max_length = 4096, truncation = True).to(device)
        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "labels": torch.tensor(all_labels),
            "ids": torch.tensor(all_ids)}

def get_standard_dataloader(data, tokenizer, shuffle, args):
    dataset = StandardDataset(data)
    collator = StandardCollator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(45)
    return DataLoader(dataset, 
                      collate_fn=collator, 
                      batch_size=args.batch_size, 
                      shuffle=shuffle, 
                      num_workers=0, 
                      worker_init_fn=seed_worker,
                      generator=generator)
            
class Bert_Model(nn.Module):
    def __init__(self, model_name, hid_dim = 768):
        super().__init__()
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name, max_position_embeddings=4098)
        print(f"number of parameters in {model_name}:", self.model.num_parameters())
        # self.linear_1 = nn.Linear(hid_dim, hid_dim // 2)
        # self.linear_2 = nn.Linear(hid_dim // 2, 1)
        self.linear = nn.Linear(hid_dim, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, batch):
        input_ids, attn_masks, labels, ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device), batch['ids'].to(device)
        print("input_id shape:", input_ids.shape)
        trans_out = self.model(input_ids = input_ids, attention_mask = attn_masks)
        # trans_cls = trans_out['last_hidden_state'][:, 0, :]
        trans_cls = trans_out["pooler_output"]
        # x = self.linear_1(trans_cls)
        # x = F.leaky_relu(x)
        # predictions = self.linear_2(x).squeeze(-1)
        predictions = self.linear(trans_cls).squeeze(-1)
        return self.loss_fn(predictions, labels), predictions, labels, ids
    
def train(train_dataloader, val_dataloader, model, args, save_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = math.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        total_train_loss = 0
        total_val_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            loss, batch_predictions, batch_labels, _ = model(batch)
            total_train_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for batch in tqdm(val_dataloader):
                loss, batch_predictions, batch_labels, _ = model(batch)
                predictions.extend(batch_predictions.tolist())
                labels.extend(batch_labels.tolist())

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = model.loss_fn(torch.tensor(predictions), torch.tensor(labels)).item()
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model_copy = copy.deepcopy(model)
            torch.save(model_copy.state_dict(), save_name)
            best_val_loss = avg_val_loss
            best_epoch = epoch
        
        if args.wandb:
            wandb.log({"train_loss": avg_train_loss, 
                       "val_loss": avg_val_loss,
                       "best epoch": best_epoch})
            
def test(test_dataloader, model):
    predictions = []
    labels = []
    question_ids = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            loss, batch_predictions, batch_labels, ids = model(batch)
            predictions.extend(batch_predictions.tolist())
            labels.extend(batch_labels.tolist())
            question_ids.extend(ids.tolist())
    print("max label:", max(labels))
    print("min label:", min(labels))    
    print("max prediction:", max(predictions))
    print("min prediction:", min(predictions))
    avg_loss = model.loss_fn(torch.tensor(predictions), torch.tensor(labels)).item()
    print(f"Test Loss: {avg_loss:.4f}")
    gt_pair_coomparisons = np.array(count_greater_pairs(labels))
    pred_pair_coomparisons = np.array(count_greater_pairs(predictions))
    # calculate the number of times the predicted one matches the ground truth
    correct_pairs_ratio = np.sum(gt_pair_coomparisons == pred_pair_coomparisons) / len(gt_pair_coomparisons)
    print("rato of correct pair comparisons:", correct_pairs_ratio)
    # calculate r2 score
    r2 = calculate_r2(np.array(labels), np.array(predictions))
    print("R^2 score:", r2) 
    # draw(labels, "ground truth graph")
    # draw(predictions, "FT predicted difficulty graph")
    # save predictions and labels in a json file
    with open("without_reasoning_LLM_predictions.json", "w") as f:
    # with open("LLM_predictions.json", "w") as f:
        json.dump({"predictions": predictions, "labels": labels, "question_ids": question_ids}, f, indent=4)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of model to save for training or load for testing", default="allenai/longformer-base-4096") # "FacebookAI/roberta-base"  "allenai/longformer-base-4096"
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--wandb', action='store_true', help='use wandb')

    args = parser.parse_args()
    
    if args.wandb:
        # wandb.init(project="MAPT_results",
        wandb.init(project="EEDI_results",
                #    name = f"FT_{args.lr}_{args.epochs}_{args.fold}",
                   name = f"without_reasoning_FT_{args.lr}_{args.epochs}_{args.fold}",
                   config=args, 
                   entity="ml4ed")
            
    # with open("manual_math_feature_data_v4.json", "r") as f:
    #     questions = json.load(f)
    # with open("math_reasoning_data.json", "r") as f:
    #     questions = json.load(f)
    with open("eedi_math_reasoning_data_with_selection_counts.json", "r") as f:
        questions = json.load(f)
    np.random.shuffle(questions)
    
    # split the data into 5 folds
    split_point = int((args.fold / 5) * len(questions))
    questions = questions[split_point:] + questions[:split_point]
    
    train_data, val_data, test_data = get_data_for_LLM(questions)
    print("number of training data:", len(train_data))
    print("number of validation data:", len(val_data))
    print("number of test data:", len(test_data))
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = LongformerTokenizer.from_pretrained(args.model_name)
    train_dataloader = get_standard_dataloader(train_data, tokenizer, True, args)
    val_dataloader = get_standard_dataloader(val_data, tokenizer, False, args)
    test_dataloader = get_standard_dataloader(test_data, tokenizer, False, args)
    # save_name = f"./ft_saved_models/{args.model_name.split('/')[1]}_{args.lr}_{args.epochs}_{args.fold}"
    # save_name = f"./ft_saved_models/eedi_{args.model_name.split('/')[1]}_{args.lr}_{args.epochs}_{args.fold}"
    save_name = f"./ft_saved_models/without_reasoning_eedi_{args.model_name.split('/')[1]}_{args.lr}_{args.epochs}_{args.fold}"

    
    if args.test:
        trained_model = Bert_Model(args.model_name).to(device)
        trained_model.load_state_dict(torch.load(save_name)) 
        test(test_dataloader, trained_model)
        # file_save_name = f"finetune"
        # get_predictions(test_dataloader, gt_data, trained_model, file_save_name)         
    else:    
        train(train_dataloader, val_dataloader, Bert_Model(args.model_name).to(device), args, save_name)
        

if __name__ == "__main__":
    main()