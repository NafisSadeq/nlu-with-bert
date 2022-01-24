import torch
from torch import nn
from transformers import BertModel


class JointBERT(nn.Module):
    def __init__(self, model_config, device,tag_dim, slot_dim, intent_dim, intent_weight=None):
        super(JointBERT, self).__init__()
        self.tag_num_labels = tag_dim
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.slot_weight = torch.tensor([1.]*slot_dim)
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier_seq = nn.Linear(self.hidden_units, self.tag_num_labels)
                self.slot_classifier_cls = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden_seq = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden_cls = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier_seq = nn.Linear(self.hidden_units, self.tag_num_labels)
                self.slot_classifier_cls = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden_seq = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden_cls = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden_seq.weight)
            nn.init.xavier_uniform_(self.slot_hidden_cls.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier_seq = nn.Linear(2 * self.bert.config.hidden_size, self.tag_num_labels)
                self.slot_classifier_cls = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier_seq = nn.Linear(self.bert.config.hidden_size, self.tag_num_labels)
                self.slot_classifier_cls = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier_seq.weight)
        nn.init.xavier_uniform_(self.slot_classifier_cls.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_seq_fct = torch.nn.CrossEntropyLoss()
        self.slot_loss_cls_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.slot_weight)

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None,slot_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden_seq(self.dropout(sequence_output)))
            pooled_output_slot = nn.functional.relu(self.slot_hidden_cls(self.dropout(pooled_output)))
            pooled_output_intent = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits_seq = self.slot_classifier_seq(sequence_output)
        outputs = (slot_logits_seq,)

        pooled_output_slot = self.dropout(pooled_output_slot)
        slot_logits_cls = self.slot_classifier_cls(pooled_output_slot)
        outputs = outputs + (slot_logits_cls,)

        pooled_output_intent = self.dropout(pooled_output_intent)
        intent_logits = self.intent_classifier(pooled_output_intent)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits_seq.view(-1, self.tag_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss_seq = self.slot_loss_seq_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss_seq,)

        if slot_tensor is not None:
            slot_loss_cls = self.slot_loss_cls_fct(slot_logits_cls, slot_tensor)
            outputs = outputs + (slot_loss_cls,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits_seq, slot_logits_cls, intent_logits, (slot_loss_seq), (slot_loss_cls), (intent_loss),
