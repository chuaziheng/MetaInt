
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, AutoConfig, AutoModel, BertModel, ElectraModel, AutoModelForMaskedLM
from collections import Counter
from transformers.modeling_outputs import SequenceClassifierOutput
import pickle

from data import device, CLS_2_WORD_DICT

def load_tokenizer():
  tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
  return tokenizer

def load_model_1():
    num_labels = len(CLS_2_WORD_DICT)
    model = MetaphorModel(num_labels, model_name='roberta', device=device, mode='test').to(device)
    if device == torch.device("cpu"):
      model.load_state_dict(torch.load('models/model_task1.pt', map_location=torch.device('cpu')))
    else:
      model.load_state_dict(torch.load('models/model_task1.pt'))

    return model

def load_model_2():
  return AutoModelForMaskedLM.from_pretrained('models/model_task2')


class MetaphorModel(nn.Module):
  def __init__(self, num_labels, model_name='roberta', last_hidden_state_dim=768, dropout_prob=0.1, device='cpu', mode='train'):
    super(MetaphorModel,self).__init__()
    self.num_labels = num_labels
    if model_name == 'roberta':
      print('initialising roberta model')
      self.model = RobertaModel.from_pretrained("roberta-base")
    elif model_name == 'bert':
      print('initialising bert model')
      self.model = BertModel.from_pretrained("bert-base-uncased")
    elif model_name == 'electra':
      print('initialising electra model')
      self.model = ElectraModel.from_pretrained("google/electra-base-discriminator")
    self.dropout = nn.Dropout(dropout_prob)
    self.classifier = nn.Linear(last_hidden_state_dim, self.num_labels) # load and initialize weights
    self.device = device
    self.mode = mode

  def forward(self, input_ids=None, attention_mask=None, metaphor_mask=None, labels=None, posneg_labels=None):
    # extract last hidden state of roberta
    last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

    # extract roberta hidden state for metaphor tokens only
    res = torch.where(metaphor_mask)
    res = [i.cpu().numpy() for i in res]
    num_tok_per_sample_counter = Counter(res[0])

    # creating new metaphor only embedding
    processed_embeddings = torch.zeros((last_hidden_state.size()[0], 768), requires_grad=True).to(self.device)
    for idx, i in enumerate(res[0]):
        processed_embeddings[i] = torch.add(processed_embeddings[i], last_hidden_state[i, res[1][idx]]/num_tok_per_sample_counter[i])

    processed_embeddings = self.dropout(processed_embeddings)

    if self.mode == 'train':
      logits = self.classifier(processed_embeddings)
      labels = labels.type(torch.LongTensor).to(device)
    else:
      old_logits = self.classifier(processed_embeddings)
      logits = torch.zeros(old_logits.size()[0], 4).to(device)  # bs, 4
      for idx, i in enumerate(posneg_labels):
        logits[idx, :] = old_logits[idx, i]
      labels = torch.zeros(old_logits.size()[0]).type(torch.LongTensor).to(device)  # label 0 will be the correct one

    loss = None

    if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

    return SequenceClassifierOutput(loss=loss, logits=logits)
