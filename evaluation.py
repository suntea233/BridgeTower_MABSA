import torch
import numpy as np


def evaluate(test_dataloader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predictions, true_labels = [], []

    for input_ids, attention_mask, pixel_values, pixel_mask, labels in test_dataloader:
        input_ids, attention_mask, pixel_values, pixel_mask, labels = input_ids.to(device), attention_mask.to(
            device), pixel_values.to(device), pixel_mask.to(device), labels.to(device)

        logits = model(input_ids, attention_mask, pixel_values, pixel_mask, labels=None)

        batch_size = input_ids.shape[0]
        if model.attention_mode == "concat_attention":
            crf_mask = torch.cat([attention_mask, torch.zeros(size=(batch_size, model.patch_len), device=device)],
                                 dim=1)
            crf_mask = crf_mask.eq(1)
        else:
            crf_mask = attention_mask.eq(1)

        logits = model.crf.decode(logits, mask=crf_mask)
        logits = np.array(logits)
        logits = logits.tolist()[0]
        predictions += logits[1:-1]  # remove ['CLS'] and ['SEP']

        true_labels += labels.tolist()[0][:len(logits)][1:-1]

    true, pre = true_labels, predictions

    label2id = {
        "O": 0,
        "B-POS": 1,
        "B-NEG": 2,
        "B-NEU": 3,
        "I": 4
    }
    id2label = {value: key for key, value in label2id.items()}

    true = [id2label[i] for i in true]
    pre = [id2label[i] for i in pre]

    tags = [("B-POS", "I"), ("B-NEG", "I"), ("B-NEU", "I")]

    def get_ans(pre, tag):
        ans = []
        start = 0
        temp = ""
        for i in range(1, len(pre) - 1):

            if pre[i] in ["B-POS", "B-NEU", "B-NEG"]:
                temp = pre[i]
            if pre[i] == tag[0]:
                start = i
            if pre[i] == tag[0] and pre[i + 1] != tag[1]:
                end = i
                ans.append([start, end])

            if pre[i] == tag[1] and pre[i + 1] != tag[1] and temp == tag[0] and pre[i - 1] != "O":
                end = i
                ans.append([start, end])

        return ans

    def f1_score(precision, recall):
        return (2 * precision * recall) / (precision + recall)

    cnt = 0
    predict = 0
    gold = 0
    for tag in tags:
        p = get_ans(pre, tag)

        predict += len(p)
        t = get_ans(true, tag)

        gold += len(t)
        for i in range(len(p)):
            for j in range(len(t)):
                if p[i][0] in t[j]:
                    if p[i][0] == t[j][0] and p[i][1] == t[j][1]:
                        cnt += 1
    if predict == 0:
        return 0
    precision = cnt / predict
    recall = cnt / gold
    f1 = f1_score(precision, recall)

    return f1,precision,recall
