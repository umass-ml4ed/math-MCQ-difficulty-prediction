import json
import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModel, AutoTokenizer
from transformers import LongformerModel, LongformerTokenizer
import argparse
from utils import initialize_seeds, get_data_for_LLM, count_greater_pairs, calculate_r2, seed_worker, check_correlation
import numpy as np
from tqdm import tqdm
import copy
import math
import wandb
import pandas as pd

device = torch.device('cuda:0')
initialize_seeds(45)
 
class StandardDataset(Dataset):
    def __init__(self, questions):
        self.data = [question for question in questions]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
    
class StandardCollator:
    # def __init__(self, tokenizer: AutoTokenizer):
    def __init__(self, tokenizer: LongformerTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        all_prompts = []
        all_labels = [sample["GroundTruthDifficulty"] for sample in batch]
        all_dis = [sample["selection_percentage"] for sample in batch]
        all_ids = [sample["ID"] for sample in batch]
        for sample in batch:
            correct_answer_prompt = (
                f"The question is: {sample['QuestionStem']}\n"
                f"The reasoning steps to reach the correct answer are: {sample['CorrectAnswerReasoning']}\n"
                f"The correct answer is: {sample['CorrectAnswer']}")
            all_prompts.append(correct_answer_prompt)
            
            student1_answer_prompt = (
                f"The question is: {sample['QuestionStem']}\n"
                f"The potential way to reach the first wrong answer is: {sample['Student1Reasoning']}\n"
                f"The first wrong answer is: {sample['Student1Answer']}")
                # f"The first wrong answer is: {sample['Student1Answer']}")
            all_prompts.append(student1_answer_prompt)
        
            student2_answer_prompt = (
                f"The question is: {sample['QuestionStem']}\n"
                f"The potential way to reach the second wrong answer is: {sample['Student2Reasoning']}\n"
                f"The second wrong answer is: {sample['Student2Answer']}")
                # f"The second wrong answer is: {sample['Student1Answer']}")
            all_prompts.append(student2_answer_prompt)
            
            student3_answer_prompt = (
                f"The question is: {sample['QuestionStem']}\n"
                f"The potential way to reach the third wrong answer is: {sample['Student3Reasoning']}\n"
                f"The third wrong answer is: {sample['Student3Answer']}")
                # f"The third wrong answer is: {sample['Student1Answer']}")
            all_prompts.append(student3_answer_prompt)
                
        # prompts_tokenized = self.tokenizer(all_prompts, padding='longest', return_tensors='pt', max_length = 512, truncation = True).to(device)
        prompts_tokenized = self.tokenizer(all_prompts, padding='longest', return_tensors='pt', max_length = 4096, truncation = True).to(device)

        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "labels": torch.tensor(all_labels),
            "dis": torch.tensor(all_dis),
            "ids": torch.tensor(all_ids)}
        # return (all_prompts, all_labels)

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
    def __init__(self, model_name, q_dim, s_dim, num_train_students, num_val_students, num_test_students, use_bilinear, CE_hyper):
        super().__init__()
        self.q_dim = q_dim
        self.s_dim = s_dim
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name)
        self.use_bilinear = use_bilinear
        self.CE_hyper = CE_hyper
        
        if self.use_bilinear:
            print("using bilinear")
            self.bilinear = nn.Parameter(torch.empty(self.s_dim, self.q_dim))
            nn.init.xavier_normal_(self.bilinear)
        else:
            print("using MLP")
            # self.linear5 = nn.Linear(s_dim, q_dim // 8)
            # self.linear6 = nn.Linear(q_dim // 8, q_dim // 4)
            # self.linear7 = nn.Linear(q_dim // 4, q_dim // 2)
            # self.linear8 = nn.Linear(q_dim // 2, q_dim)
            self.bilinear_5 = nn.Parameter(torch.empty(self.s_dim, self.q_dim // 4))
            nn.init.xavier_normal_(self.bilinear_5)
            self.bilinear_6 = nn.Parameter(torch.empty(self.q_dim // 4, self.q_dim // 1))
            nn.init.xavier_normal_(self.bilinear_6)
            # self.bilinear_7 = nn.Parameter(torch.empty(self.q_dim // 2, self.q_dim))
            # nn.init.xavier_normal_(self.bilinear_7)

        self.linear_1 = nn.Linear(4, 16)
        self.linear_2 = nn.Linear(16, 8)
        self.linear_3 = nn.Linear(8, 4)
        self.linear_4 = nn.Linear(4, 1)
        
        self.loss_fn = nn.MSELoss()

        mean = torch.zeros(self.s_dim)
        covariance_matrix = torch.eye(self.s_dim)
        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        self.students = mvn.sample((num_train_students,)).to(device)
        print("students shape:", self.students.shape)
        
        # # save students to a json file
        # students_dict = self.students.tolist()
        # with open("students_knowledge_level.json", "w") as f:
        #     json.dump(students_dict, f, indent=4)

        # import pdb; pdb.set_trace()
        
        # self.train_students = mvn.sample((num_train_students,)).to(device)
        # print("train students shape:", self.train_students.shape)
        # self.val_students = mvn.sample((num_val_students,)).to(device)
        # print("val students shape:", self.val_students.shape)
        # self.test_students = mvn.sample((num_test_students,)).to(device)
        # print("test students shape:", self.test_students.shape)
        
        
    def forward(self, batch, test = False, val = False):
        # sentences = batch[0]
        # for sentence in sentences:
        #     print(sentence)
        #     print('_' * 50)
        # labels = batch[1]
        # print("labels:", labels)
        # import pdb; pdb.set_trace()
        
        input_ids, attn_masks, labels, dis, ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device), batch['dis'].to(device), batch['ids'].to(device)
        print("input_ids shape:", input_ids.shape)
        # print("attn_masks shape:", attn_masks.shape)
        # print("labels shape:", labels.shape)
        trans_out = self.model(input_ids = input_ids, attention_mask = attn_masks)
        # trans_cls = trans_out['last_hidden_state'][:, 0, :]
        trans_cls = trans_out["pooler_output"]
        # print("trans_cls shape:", trans_cls.shape)
        if test:
            students = self.students
            # students = self.test_students
        elif val:
            students = self.students
            # students = self.val_students
        else:
            students = self.students
            # students = self.train_students

        if self.use_bilinear:
            logits = students @ self.bilinear @ trans_cls.T
        else:
            # x = self.linear5(students)
            # x = F.leaky_relu(x)
            # x = self.linear6(x)
            # x = F.leaky_relu(x)
            # x = self.linear7(x)
            # x = F.leaky_relu(x)
            # x = self.linear8(x)
            # x = F.leaky_relu(x)
            x = students @ self.bilinear_5
            x = F.leaky_relu(x)
            x = x @ self.bilinear_6
            x = F.leaky_relu(x)
            # x = x @ self.bilinear_7
            # x = F.leaky_relu(x)
            logits = x @ trans_cls.T
        logits_shape = logits.shape
        # print("logits shape:", logits.shape)
        # print("logits values:", logits)
        logits = logits.view(-1, 4)
        # print("logits shape after reshape:", logits.shape)
        # print("logits values after reshape:", logits)
        
        logits = torch.softmax(logits, dim = 1)
        # print("logits shape after softmax:", logits.shape)
        # print("logits values after softmax:", logits)
        logits = logits.view(logits_shape)
        # print("logits shape after reshape:", logits.shape)
        # print("logits values after reshape:", logits)
        logits = torch.mean(logits, dim = 0)
        # print("logits shape after mean:", logits.shape)
        # print("logits values after mean:", logits)
        logits = logits.view(-1, 4)
        x = self.linear_1(logits)
        x = F.tanh(x)
        x = self.linear_2(x)
        x = F.tanh(x)
        x = self.linear_3(x)
        x = F.tanh(x)
        predictions = self.linear_4(x).squeeze(-1)
        if test:
            loss = self.loss_fn(predictions, labels)
        elif val:
            loss = self.loss_fn(predictions, labels)
        else:
            # write the loss function here
            log_logits = torch.log(logits)
            KL_loss = F.kl_div(log_logits, dis, reduction='batchmean')
            loss = self.loss_fn(predictions, labels) + self.CE_hyper * KL_loss
        # print("predictions shape:", predictions.shape)
        # print("labels shape:", labels.shape)
        # calculate entropy of each logits
        entropy = -torch.sum(logits * torch.log(logits), dim = 1)
        return loss, predictions, labels, logits, entropy, ids
    
    def calcualte_student_difficulty(self, batch):
        input_ids, attn_masks, ids = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['ids'].to(device)
        trans_out = self.model(input_ids = input_ids, attention_mask = attn_masks)
        trans_cls = trans_out["pooler_output"]
        students = self.students
        x = students @ self.bilinear_5
        x = F.leaky_relu(x)
        x = x @ self.bilinear_6
        x = F.leaky_relu(x)
        logits = x @ trans_cls.T
        logits = torch.softmax(logits, dim = 1)
        x = self.linear_1(logits)
        x = F.tanh(x)
        x = self.linear_2(x)
        x = F.tanh(x)
        x = self.linear_3(x)
        x = F.tanh(x)
        predictions = self.linear_4(x).squeeze(-1)
        return predictions
        
    
def train(train_dataloader, val_dataloader, model, args, save_name):
    if args.use_bilinear:
        optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.LLM_lr},
        {'params': model.bilinear, 'lr': args.interaction_lr},
        {'params': model.linear_1.parameters(), 'lr': args.MLP_lr},
        {'params': model.linear_2.parameters(), 'lr': args.MLP_lr},
        {'params': model.linear_3.parameters(), 'lr': args.MLP_lr},
        {'params': model.linear_4.parameters(), 'lr': args.MLP_lr}], weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': args.LLM_lr},
        # {'params': model.linear5.parameters(), 'lr': args.MLP_lr},
        # {'params': model.linear6.parameters(), 'lr': args.MLP_lr},
        # {'params': model.linear7.parameters(), 'lr': args.MLP_lr},
        # {'params': model.linear8.parameters(), 'lr': args.MLP_lr},
        {'params': model.bilinear_5, 'lr': args.interaction_lr},
        {'params': model.bilinear_6, 'lr': args.interaction_lr},
        # {'params': model.bilinear_7, 'lr': args.interaction_lr},
        {'params': model.linear_1.parameters(), 'lr': args.MLP_lr},
        {'params': model.linear_2.parameters(), 'lr': args.MLP_lr},
        {'params': model.linear_3.parameters(), 'lr': args.MLP_lr},
        {'params': model.linear_4.parameters(), 'lr': args.MLP_lr}], weight_decay=args.wd)

    best_val_loss = math.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            loss, _, _, _, _,_ = model(batch, test = False, val = False)
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
                val_loss, batch_predictions, batch_labels, batch_logits, batch_entropy, _ = model(batch, test = False, val = True)
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
            
def test(test_dataloader, model, save_name):
    model.eval()
    with torch.no_grad():
        predictions = []
        labels = []
        logits = []
        entropys = []
        question_ids = []
        for batch in tqdm(test_dataloader):
            loss, batch_predictions, batch_labels, batch_logits, batch_entropy, ids = model(batch, test = True, val = False)
            predictions.extend(batch_predictions.tolist())
            labels.extend(batch_labels.tolist())
            logits.extend(batch_logits.tolist())
            entropys.extend(batch_entropy.tolist())
            question_ids.extend(ids.tolist())
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
    check_correlation(predictions, labels, logits, entropys)
    
    # # save predictions, labels amd logits into a csv file using pandas
    # df = pd.DataFrame({"predictions": predictions, "labels": labels, "logits": logits, "entropys": entropys})
    # df.to_csv(f"saved_results/{save_name}.csv", index=False) 
    with open("our_method.json", "w") as f:
        json.dump({"predictions": predictions, "labels": labels, "question_ids": question_ids}, f, indent=4)
    
def students_prediction(test_data_loader, model):
    model.eval()
    with torch.no_grad():
        predictions = []
        ids = []
        for batch in tqdm(test_data_loader):
            prediction = model.calcualte_student_difficulty(batch)
            # should keep the prediction shape in predictions
            predictions.append(prediction.tolist())
            ids.extend(batch['ids'].tolist())
    # save as a dictionary
    predictions_dict = dict(zip(ids, predictions))
    with open("students_predictions.json", "w") as f:
        json.dump(predictions_dict, f, indent=4)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of model to save for training or load for testing", default="allenai/longformer-base-4096") #  "FacebookAI/roberta-base" "allenai/longformer-base-4096"
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--q_dim", type=int, default=768)
    parser.add_argument("--s_dim", type=int, default=1)
    parser.add_argument("--LLM_lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--MLP_lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--interaction_lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--CE_hyper", type=float, default=1e-1, help="CE hyper-params")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num_train_student", type=int, default=100, help="number of train students")
    parser.add_argument("--num_val_student", type=int, default=100, help="number of val students")
    parser.add_argument("--num_test_student", type=int, default=100, help="number of test students")
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--use_bilinear', action='store_true', help='use bilinear')
    

    args = parser.parse_args()
    
    # if args.wandb:
    #     wandb.init(project="MAPT_cleaning_24",
    #                name = f"entropy_{args.model_name.split('/')[1]}_{args.LLM_lr}_{args.MLP_lr}_{args.interaction_lr}_{args.epochs}_{args.fold}_num_train_student_{args.num_train_student}_num_val_student_{args.num_val_student}_s_dim_{args.s_dim}_use_bilinear_{args.use_bilinear}_CE_hyper_{args.CE_hyper}",
    #                config=args, 
    #                entity="ml4ed")
    
    if args.wandb:
        wandb.init(project="EEDI_results",
                   name = f"entropy_{args.model_name.split('/')[1]}_{args.LLM_lr}_{args.MLP_lr}_{args.interaction_lr}_{args.epochs}_{args.fold}_num_train_student_{args.num_train_student}_num_val_student_{args.num_val_student}_s_dim_{args.s_dim}_use_bilinear_{args.use_bilinear}_CE_hyper_{args.CE_hyper}",
                   config=args, 
                   entity="ml4ed")
            
    # with open("math_reasoning_data_with_selection_counts.json", "r") as f:
    #     data = json.load(f)
    with open("eedi_math_reasoning_data_with_selection_counts.json", "r") as f:
        data = json.load(f)
    np.random.shuffle(data)
    
    # split the data into 5 folds
    split_point = int((args.fold / 5) * len(data))
    data = data[split_point:] + data[:split_point]
    
    # difficutlies = [sample["GroundTruthDifficulty"] for sample in data]
    # draw(difficutlies, "ground_truth_difficulty")
    # import pdb; pdb.set_trace()
    
    train_data, val_data, test_data = get_data_for_LLM(data)
    print("train data length:", len(train_data))
    print("val data length:", len(val_data))
    print("test data length:", len(test_data))
    
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = LongformerTokenizer.from_pretrained(args.model_name)
    train_dataloader = get_standard_dataloader(train_data, tokenizer, True, args)
    val_dataloader = get_standard_dataloader(val_data, tokenizer, False, args)
    test_dataloader = get_standard_dataloader(test_data, tokenizer, False, args)
    # save_name = f"./entropy_saved_models/new_idea_{args.model_name.split('/')[1]}_{args.LLM_lr}_{args.MLP_lr}_{args.interaction_lr}_{args.epochs}_{args.fold}_num_train_student_{args.num_train_student}_num_val_student_{args.num_val_student}_s_dim_{args.s_dim}_use_bilinear_{args.use_bilinear}_CE_hyper_{args.CE_hyper}"
    save_name = f"./entropy_saved_models/eedi_new_idea_{args.model_name.split('/')[1]}_{args.LLM_lr}_{args.MLP_lr}_{args.interaction_lr}_{args.epochs}_{args.fold}_num_train_student_{args.num_train_student}_num_val_student_{args.num_val_student}_s_dim_{args.s_dim}_use_bilinear_{args.use_bilinear}_CE_hyper_{args.CE_hyper}"

    print("save name:", save_name)

    model = Bert_Model(args.model_name, args.q_dim, args.s_dim, args.num_train_student, args.num_val_student, args.num_test_student, args.use_bilinear, args.CE_hyper).to(device)
    
    model.load_state_dict(torch.load(save_name))
    students_prediction(test_dataloader, model)
    
    
    
    # if args.test:
    #     # print(model.parameters)
    #     # print(model.model.encoder.layer[-1].output.dense)
    #     # print("before training roberta:", list(model.model.encoder.layer[-1].output.dense.parameters()))
    #     # print("before training linear1:", list(model.linear_1.parameters()))
    #     model.load_state_dict(torch.load(save_name))
    #     # print("after training roberta:", list(model.model.encoder.layer[-1].output.dense.parameters()))
    #     # print("after training linear1:", list(model.linear_1.parameters()))
    #     test(test_dataloader, model, save_name.split('/')[-1])        
    # else:    
    #     train(train_dataloader, val_dataloader, model, args, save_name)
        

if __name__ == "__main__":
    main()