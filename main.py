import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model import BridgeTowerForMABSA
from dataset import TwitterDataset
from evaluation import evaluate

from tqdm import tqdm

def parse_args():
    parser =argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="Twitter2015")
    parser.add_argument("--queue_size",type=int,default=256,choices=[128,256,512,1024])
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--lr",type=float,default=5e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-2)
    parser.add_argument("--num_warmup_steps",type=int,default=10)
    parser.add_argument("--output_dir",type=str,default="output")
    parser.add_argument("--input_dim",type=int,default=768)
    parser.add_argument("--hidden_dim",type=int,default=768)
    parser.add_argument("--output_dim",type=int,default=768)
    parser.add_argument("--num_labels",type=int,default=5)
    parser.add_argument("--dropout_rate",type=float,default=0.1)
    parser.add_argument("--path",type=str,default=r"C:\uni_transformer\UniTransformer\MABSA_datasets")
    parser.add_argument("--max_grad_norm",type=float,default=1.0)
    parser.add_argument("--gpu",type=str,default="0")
    parser.add_argument("--pad_id",type=int,default=0)
    parser.add_argument("--patch_len",type=int,default=325)
    parser.add_argument("--max_len",type=int,default=50)
    parser.add_argument("--momentum",type=float,default=0.999)
    parser.add_argument("--temp",type=float,default=0.07)
    parser.add_argument("--train_batch_size",type=int,default=8)
    parser.add_argument("--eval_batch_size",type=int,default=1)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--pretrained_model",type=str,default="BridgeTower/bridgetower-base")
    parser.add_argument("--adapter_hidden_dim",type=int,default=512)
    parser.add_argument("--is_adapter",type=bool,default=False)
    parser.add_argument("--attention_mode",type=str,default='concat_attention',choices=['weighted_based_addition','cross_attention','gate_attention','concat_attention'])
    parser.add_argument("--is_attention",type=bool,default=True)


    parser.add_argument("--first_step_epochs",type=int,default=0)
    parser.add_argument("--second_step_epochs",type=int,default=20)
    parser.add_argument("--first_step_lr",type=float,default=5e-4,choices=[1e-4,3e-4,5e-4,7e-4,9e-4])
    parser.add_argument("--first_step_weight_decay",type=float,default=1e-2,choices=[1e-2,1e-3,1e-4,1e-5,1e-6])
    parser.add_argument("--second_step_lr",type=float,default=5e-5,choices=[1e-5,3e-5,5e-5,7e-5,9e-5])
    parser.add_argument("--second_step_weight_decay",type=float,default=1e-2,choices=[1e-2,1e-3,1e-4,1e-5,1e-6])
    args = parser.parse_args()
    return args


def first_step_train(args):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    train_2015_dataset = TwitterDataset(args, train=True)
    train_2015_dataloader = DataLoader(train_2015_dataset, batch_size=args.train_batch_size, collate_fn=train_2015_dataset.collate_fn,
                                       shuffle=True)


    model = BridgeTowerForMABSA(args).to(device)


    first_step_epochs = args.first_step_epochs
    first_step_lr = args.first_step_lr
    first_step_weight_decay = args.first_step_weight_decay
    num_warmup_steps = args.num_warmup_steps
    max_grad_norm = args.max_grad_norm

    first_total_steps = len(train_2015_dataset) * first_step_epochs


    first_step_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=first_step_lr, weight_decay=first_step_weight_decay)
    first_step_scheduler = get_linear_schedule_with_warmup(first_step_optimizer,
                                                           num_warmup_steps=num_warmup_steps,
                                                           num_training_steps=first_total_steps)

    for epoch in tqdm(range(first_step_epochs)):
        losses = 0
        cnt = 0
        for input_ids, attention_mask,pixel_values,pixel_mask,labels in train_2015_dataloader:
            cnt += 1
            model.train()

            input_ids, attention_mask,pixel_values,pixel_mask,labels = input_ids.to(device), attention_mask.to(device),pixel_values.to(device),pixel_mask.to(device),labels.to(device)
            cl_loss = model(input_ids,attention_mask,pixel_values,pixel_mask,labels=None)

            first_step_optimizer.zero_grad()
            cl_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            first_step_optimizer.step()
            first_step_scheduler.step()
            losses += cl_loss.item()

        print("cl_loss = {}".format(losses / cnt))

    print("first_step finished")

    return model


def second_step_train(args,model):

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    train_2015_dataset = TwitterDataset(args, train=True)
    train_2015_dataloader = DataLoader(train_2015_dataset, batch_size=args.train_batch_size, collate_fn=train_2015_dataset.collate_fn,
                                       shuffle=True)

    test_2015_dataset = TwitterDataset(args, train=False)
    test_2015_dataloader = DataLoader(test_2015_dataset, batch_size=args.eval_batch_size, collate_fn=test_2015_dataset.collate_fn,
                                      shuffle=True)

    num_warmup_steps = args.num_warmup_steps
    max_grad_norm = args.max_grad_norm

    model.two_stage = True

    second_step_lr = args.second_step_lr
    second_step_weight_decay = args.second_step_weight_decay
    second_step_epochs = args.second_step_epochs

    second_total_steps = len(train_2015_dataset) * second_step_epochs

    second_step_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=second_step_lr,
                                              weight_decay=second_step_weight_decay)
    second_step_scheduler = get_linear_schedule_with_warmup(second_step_optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=second_total_steps)

    for epoch in tqdm(range(second_step_epochs)):
        losses = 0
        cnt = 0
        for input_ids, attention_mask,pixel_values,pixel_mask,labels in train_2015_dataloader:
            model.train()
            cnt += 1

            input_ids, attention_mask, pixel_values, pixel_mask, labels = input_ids.to(device), attention_mask.to(
                device), pixel_values.to(device), pixel_mask.to(device), labels.to(device)

            crf_loss = model(input_ids,attention_mask,pixel_values,pixel_mask,labels=labels)

            second_step_optimizer.zero_grad()
            crf_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            second_step_optimizer.step()
            second_step_scheduler.step()
            losses += crf_loss.item()


        f1,p,r = evaluate(test_2015_dataloader, model)
        print(f1,p,r)

        if args.dataset == "Twitter2015":
            sota = 0.688
        else:
            sota = 0.704

        if f1 >= sota:
            torch.save(model.state_dict(), "best_model.pt")

        print("crf_loss = {}".format(losses / cnt))


if __name__ == "__main__":

    args = parse_args()
    print("args", args)
    torch.manual_seed(args.seed)

    model = first_step_train(args)

    # for attention_mode in ['weighted_based_addition','cross_attention','gate_attention','concat_attention']:

    for weight_decay in [1e-2,1e-3,1e-4,1e-5,1e-6]:
        # args.attention_mode = attention_mode
        args.weight_decay = weight_decay
        second_step_train(args,model)