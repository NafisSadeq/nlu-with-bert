import torch
import numpy as np

from torch import nn
from tqdm import tqdm


class BiLSTM(nn.Module):
    def __init__(self, model_config, device, slot_dim, intent_dim, intent_weight=None):
        super(BiLSTM, self).__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        pretrained_embeddings = [list(map(float, l.split()[1:])) for l in tqdm(open(model_config['pretrained_weights'], 'r'), desc=f"load {model_config['pretrained_weights']} as pretrained embeddings:")]
        pretrained_embeddings = [[0]*len(pretrained_embeddings[-1])] + pretrained_embeddings + [[0]*len(pretrained_embeddings[-1])] # padding: 0 and unk: -1
        self.emb = nn.Embedding.from_pretrained(torch.Tensor(pretrained_embeddings), freeze=False, padding_idx=0)  

        self.seq = nn.LSTM(
            input_size = model_config['embed_size'],
            hidden_size = model_config['hidden_units'],
            num_layers = model_config['num_layers'],
            bidirectional = True,
            batch_first = True
        )

        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(4 * self.hidden_units, self.hidden_units)
                self.slot_hidden = nn.Linear(4 * self.hidden_units, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(2 * self.hidden_units, self.hidden_units)
                self.slot_hidden = nn.Linear(2 * self.hidden_units, self.hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * model_config['embed_size'], self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * model_config['embed_size'], self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(model_config['embed_size'], self.intent_num_labels)
                self.slot_classifier = nn.Linear(model_config['embed_size'], self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor=None, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):

        last_idx = (word_seq_tensor != 0).sum(dim=-1) - 1

        word_emb = self.emb(word_seq_tensor)
        outputs, _ = self.seq(input=word_emb)

        sequence_output = outputs
        pooled_output = outputs[torch.arange(outputs.size(0)), last_idx]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.seq(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.seq(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
            pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits, )

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, (slot_loss), (intent_loss),
