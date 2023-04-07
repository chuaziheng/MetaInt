from transformers import AutoTokenizer, RobertaModel, AutoConfig, AutoModel, BertModel, ElectraModel
import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from data import collate_batch_task1, collate_batch_task2, process_inputs, process_inputs_task2, device
from model import MetaphorModel

def conduct_metaphor_interpretation_task1(model, tokenizer, input_data={}):

    example_input = process_inputs(input_data)

    example_input = collate_batch_task1([example_input], tokenizer)
    model.eval()
    model.to(device)
    # all_proba = []
    # all_preds = []


    # metric = evaluate.load("accuracy")
    example_input = {k: v.to(device) for k, v in example_input.items()}
    with torch.no_grad():
        outputs = model(input_ids=example_input['input_ids'], attention_mask=example_input['attention_mask'], metaphor_mask=example_input['metaphor_mask'], labels=example_input['labels'], posneg_labels=example_input['posneg_labels'])

    predictions = torch.argmax(outputs.logits, dim=-1)
    all_proba = nn.Softmax(dim=1)(outputs.logits).tolist()
    all_preds = predictions.tolist()
    return all_proba[0]

def conduct_metaphor_interpretation_task2(model, tokenizer, input_data={}):
    example_input_raw = process_inputs_task2(input_data, tokenizer)

    decoded_input_sentences = [i['decoded_input_sentence'] for i in example_input_raw]
    print(example_input_raw)

    example_input = collate_batch_task2(example_input_raw, tokenizer)

    model.eval()
    model.to(device)
    example_input['input_ids'] = example_input['input_ids'].to(device)
    example_input['attention_mask'] = example_input['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=example_input['input_ids'], attention_mask=example_input['attention_mask'])

    logits = outputs.logits
    proba = nn.Softmax(dim=1)(logits)
    mask_token_index = np.argwhere(example_input["input_ids"].cpu() == tokenizer.mask_token_id)
    mask_token_index = mask_token_index.tolist()
    pred_logits = get_mask_logits(proba, mask_token_index)
    label_dict = {k:v for k, v in example_input.items() if k in ['label_tok', 'neg_label_1_tok', 'neg_label_2_tok', 'neg_label_3_tok']}
    results = get_label_probability(pred_logits,label_dict)
    preds = np.argmax(np.array(results), axis=1)

    final_preds = list(np.mean(results, axis=0))
    return final_preds, decoded_input_sentences



def get_label_probability(pred_logits, label_dict):
    label_names = label_dict.keys()
    res = []
    for i in range(pred_logits.size()[0]):  # for each datapoint
        sub_res = []
        for name in label_names:
            label_tok_list = label_dict[name][i]
            sub_res.append(float(pred_logits[i, label_tok_list].mean()))
        res.append(sub_res)
    return res


def get_mask_logits(proba, mask_token_index):
    pred_proba = torch.zeros(proba.size()[0], proba.size()[2])
    for idx, (r, c) in enumerate(zip(*mask_token_index)):
        pred_proba[idx] = proba[r, c, :].cpu()
    return pred_proba
