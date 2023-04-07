import torch
import pickle
from typing import List

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

INPUT_EXAMPLES_DICT = {
    'Custom': {
        'metaphor_sentence': "",
        'positive': "",
        'neg_1': "",
        'neg_2': "",
        'neg_3': "",
    },
    'Example 1': {
        'metaphor_sentence': "Now that they ca n't X_get X_hold of cocaine they 'll just crash out .",
        'positive': "acquire",
        'neg_1': "achieve",
        'neg_2': "collect",
        'neg_3': "earn",
    },
    'Example 2': {
        'metaphor_sentence': "",
        'positive': "",
        'neg_1': "",
        'neg_2': "",
        'neg_3': "",
    },
    'Example 3': {
        'metaphor_sentence': "",
        'positive': "",
        'neg_1': "",
        'neg_2': "",
        'neg_3': "",
    },
}

with open('data/CLS_2_WORD_DICT_POSNEG.pkl', 'rb') as f:
    CLS_2_WORD_DICT = pickle.load(f)

WORD_2_CLS_DICT = {v: k for k, v in CLS_2_WORD_DICT.items()}

def process_inputs(input_dict):
    """
    input_dict = {
        'input_sentence_raw': input_metaphor_sentence,
        'pos_label': input_positive,
        'neg_label_1': input_neg_1,
        'neg_label_2': input_neg_2,
        'neg_label_3': input_neg_3,
    }
    """
    processed_input_dict = {}

    input_sentence = input_dict['input_sentence_raw'].split(" ")
    metaphor_mask = []
    metaphor_word = ""
    for idx in range(len(input_sentence)):
        if input_sentence[idx].startswith("M_") or input_sentence[idx].startswith("X_"):
            metaphor_mask.append(1)
            input_sentence[idx] = input_sentence[idx][2:]
        else:
            metaphor_mask.append(0)
    assert len(input_sentence) == len(metaphor_mask), "input sentence and metaphor mask not equal"
    processed_input_dict['metaphor_mask'] = metaphor_mask
    processed_input_dict['input_sentence'] = " ".join(input_sentence)
    processed_input_dict['label_idx'] = WORD_2_CLS_DICT[input_dict['pos_label']]

    posneg_idx = []
    for label in ['pos_label', 'neg_label_1', 'neg_label_2', 'neg_label_3']:
        posneg_idx.append(WORD_2_CLS_DICT[input_dict[label]])
    processed_input_dict['posneg_idx'] = posneg_idx
    return processed_input_dict


def process_inputs_task2(input_dict, tokenizer) -> List:
    input_sentence = input_dict['input_sentence_raw'].split(" ")
    metaphor_mask = []
    for idx in range(len(input_sentence)):
        if input_sentence[idx].startswith("M_") or input_sentence[idx].startswith("X_"):
            metaphor_mask.append(1)
            input_sentence[idx] = input_sentence[idx][2:]
        else:
            metaphor_mask.append(0)
    assert len(input_sentence) == len(metaphor_mask), "input sentence and metaphor mask not equal"

    # tok input sentence
    tokenized_input_sentence = tokenizer(
        input_sentence,
        is_split_into_words=True
    )


    metaphor_mask_tok = []
    word_ids = tokenized_input_sentence.word_ids()
    for word_idx in word_ids:
        if word_idx is None:
            metaphor_mask_tok.append(0)
        else:
            cur_idx = metaphor_mask[word_idx]
            metaphor_mask_tok.append(cur_idx)

    tokenized_inputs = tokenized_input_sentence['input_ids']
    attention_mask = tokenized_input_sentence['attention_mask']
    met_start, met_end = None, None
    for idx, mask in enumerate(metaphor_mask_tok):
        if mask == 1 and met_start is None:
            met_start = idx
        if mask == 0 and met_start is not None and not met_end:
            met_end = idx
        elif mask == 1 and idx == len(metaphor_mask_tok) - 1:
            met_end = idx + 1

    label_tok = tokenizer(input_dict['pos_label'].split('_'), is_split_into_words=True)['input_ids'][1:-1]
    neg_label_1_tok = tokenizer(input_dict['neg_label_1'].split('_'), is_split_into_words=True)['input_ids'][1:-1]
    neg_label_2_tok = tokenizer(input_dict['neg_label_2'].split('_'), is_split_into_words=True)['input_ids'][1:-1]
    neg_label_3_tok = tokenizer(input_dict['neg_label_3'].split('_'), is_split_into_words=True)['input_ids'][1:-1]


    batch_list = []
    attention_mask += [1, 1]

    for idx in range(met_start, met_end):
        processed_input_dict = {}
        metaphor_token = tokenized_inputs[idx]
        temp_tok_input = tokenized_inputs.copy()
        temp_tok_input[idx] = tokenizer.mask_token_id
        eos = temp_tok_input.pop()

        temp_tok_input.append(tokenizer.sep_token_id)
        temp_tok_input.append(metaphor_token)
        temp_tok_input.append(eos)



        processed_input_dict['tokenized_input_sentence'] = temp_tok_input
        processed_input_dict['decoded_input_sentence'] = tokenizer.decode(temp_tok_input)
        processed_input_dict['attention_mask'] = attention_mask

        processed_input_dict['label_tok'] = label_tok
        processed_input_dict['neg_label_1_tok'] = neg_label_1_tok
        processed_input_dict['neg_label_2_tok'] = neg_label_2_tok
        processed_input_dict['neg_label_3_tok'] = neg_label_3_tok
        batch_list.append(processed_input_dict)

    return batch_list

def collate_batch_task1(batch, tokenizer):  # gotta write custom collate_fn because idt labels can be padded using the DataCollateWithPadding library function
    """
    input_sentence: raw text
    metaphor mask: binary mask
    label_idx
    posneg_idx
    """
    tokenized_inputs = tokenizer(
        [i["input_sentence"].split() for i in batch],
        padding=True,
        truncation=True,
        return_tensors='pt',
        is_split_into_words=True
        )

    metaphor_mask = []
    for idx, i in enumerate(batch):
        if isinstance(i['metaphor_mask'], str):
            old_metaphor_mask = eval(i['metaphor_mask'])
        else:
            old_metaphor_mask = i['metaphor_mask']


        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(0)
            else:
                try:
                    cur_idx = old_metaphor_mask[word_idx]
                except:
                    print(len(old_metaphor_mask), word_idx)
                label_ids.append(cur_idx)

        assert 1 in label_ids, f"{old_metaphor_mask}, {label_ids}"
        metaphor_mask.append(label_ids)

    metaphor_mask = torch.BoolTensor(metaphor_mask)

    labels = [i["label_idx"] for i in batch]
    labels = torch.Tensor(labels)

    input_batch = tokenized_inputs
    input_batch["metaphor_mask"] = metaphor_mask
    input_batch['labels'] = labels

    if isinstance(i['posneg_idx'], str):
        posneg_labels = [eval(i["posneg_idx"]) for i in batch]
    else:
        posneg_labels = [i["posneg_idx"] for i in batch]

    posneg_labels = torch.LongTensor(posneg_labels)
    input_batch['posneg_labels'] = posneg_labels

    input_batch = { desired_key: input_batch[desired_key] for desired_key in ['input_ids', 'labels', 'attention_mask', 'metaphor_mask', 'posneg_labels'] }
    return input_batch

def collate_batch_task2(batch, tokenizer):
    input_batch = {}

    input_batch['input_ids'] = pad_sequences([i['tokenized_input_sentence'] for i in batch], dtype='input_ids')
    input_batch['attention_mask'] = pad_sequences([i['attention_mask'] for i in batch], dtype='attention_mask')


    input_batch['label_tok'] = [i['label_tok'] for i in batch]
    input_batch['neg_label_1_tok'] = [i['neg_label_1_tok'] for i in batch]
    input_batch['neg_label_2_tok'] = [i['neg_label_2_tok'] for i in batch]
    input_batch['neg_label_3_tok'] = [i['neg_label_3_tok'] for i in batch]
    return input_batch


def pad_sequences(sequence, dtype):
    max_len = max(len(x) for x in sequence)

    if dtype == 'input_ids':
        padded_seq = torch.ones(len(sequence), max_len, dtype=torch.int32)
    elif dtype == 'attention_mask':
        padded_seq = torch.zeros(len(sequence), max_len, dtype=torch.int32)


    for idx, seq in enumerate(sequence):
        padded_seq[idx, :len(seq)] = torch.Tensor(seq)
    return padded_seq  # tensor
