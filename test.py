import argparse
import os
import json
import random
import numpy as np
import torch
from dataloader import Dataloader
from jointBERT import JointBERT
from collections import OrderedDict
from postprocess import is_slot_da, calculateF1, calculateF1perIntent,calculateF1perSlotCLS, calculateF1perSlot, recover_intent, recover_slot, recover_tag

# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# arguments
parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path',
                    help='path to config file')


if __name__ == '__main__':
    # load arguments
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']

    set_seed(config['seed'])
       
    print("Data Dir:",data_dir)
    
    # load data
    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    slot_vocab = json.load(open(os.path.join(data_dir, 'slot_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
    dataloader = Dataloader(intent_vocab=intent_vocab,slot_vocab=slot_vocab, tag_vocab=tag_vocab,
                            pretrained_weights=config['model']['pretrained_weights'])
    print('intent num:', len(intent_vocab))
    print('slot num:', len(slot_vocab))
    print('tag num:', len(tag_vocab))
    for data_key in ['val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))), data_key,
                             cut_sen_len=0, use_bert_tokenizer=config['use_bert_tokenizer'])
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load model
    model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.slot_dim, dataloader.intent_dim)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
    model.to(DEVICE)
    model.eval()

    batch_size = config['model']['batch_size']

    data_key = 'test'
    predict_golden = {'intent': [], 'slot': [], 'tag': []}
    slot_loss_seq, slot_loss_cls, intent_loss = 0, 0, 0
    
    # start evaluation
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=data_key):
        # load data batch
        pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, slot_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None
        # model prediction for slots & intents
        with torch.no_grad():
            slot_logits_seq, slot_logits_cls, intent_logits, batch_slot_loss_seq, batch_slot_loss_cls, batch_intent_loss = model.forward(word_seq_tensor,
                                                                                           word_mask_tensor,
                                                                                           tag_seq_tensor,
                                                                                           tag_mask_tensor,
                                                                                           intent_tensor,
                                                                                           slot_tensor,
                                                                                           context_seq_tensor,
                                                                                           context_mask_tensor)
        # loss calculation
        slot_loss_seq += batch_slot_loss_seq.item() * real_batch_size
        slot_loss_cls += batch_slot_loss_cls.item() * real_batch_size
        intent_loss += batch_intent_loss.item() * real_batch_size
        
        # metrics calculation
        for j in range(real_batch_size):
                tag_predicts = recover_tag(dataloader, intent_logits[j], slot_logits_seq[j], tag_mask_tensor[j],
                                            ori_batch[j][0], ori_batch[j][-5])
                slot_predicts = recover_slot(dataloader, slot_logits_cls[j])
                intent_predicts = recover_intent(dataloader, intent_logits[j])
                tag_labels = ori_batch[j][3]
                intent_labels = ori_batch[j][2]

                slot_labels = set()
                for tag in ori_batch[j][1]:
                    if(tag!='O'):
                        slot="-".join(tag.split('-')[1:])
                        slot_labels.add(slot)

                slot_labels=list(slot_labels)

                predict_golden['tag'].append({
                    'predict': [x for x in tag_predicts if is_slot_da(x)],
                    'golden': [x for x in tag_labels if is_slot_da(x)]
                })

                predict_golden['slot'].append({
                    'predict': slot_predicts,
                    'golden': slot_labels
                })

                predict_golden['intent'].append({
                    'predict': intent_predicts,
                    'golden': intent_labels
                })
        print('[%d|%d] samples' % (len(predict_golden['slot']), len(dataloader.data[data_key])))

    total = len(dataloader.data[data_key])
    slot_loss_seq /= total
    slot_loss_cls /= total
    intent_loss /= total
    print('%d samples %s' % (total, data_key))
    print('\t slot loss seq:', slot_loss_seq)
    print('\t slot loss cls:', slot_loss_cls)
    print('\t intent loss:', intent_loss)

    result_list=[]
    result_list.append('slot loss seq,'+str(slot_loss_seq))
    result_list.append('slot loss cls,'+str(slot_loss_cls))
    result_list.append('intent loss,'+str(intent_loss))
    result_list.append("---,---")

    for x in ['intent', 'slot','tag']:
        precision, recall, F1 = calculateF1(predict_golden[x],x)
        print('-' * 20 + x + '-' * 20)
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

        result_list.append(x+' precision,'+str(100 * precision))
        result_list.append(x+' recall,'+str(100 * recall))
        result_list.append(x+' f1,'+str(100 * F1))
        result_list.append("---,---")

    precision, recall, F1 = calculateF1perIntent(predict_golden['intent'])
    precision = OrderedDict(sorted(precision.items(), key=lambda t: t[1],reverse=True))
    recall = OrderedDict(sorted(recall.items(), key=lambda t: t[1],reverse=True))
    F1 = OrderedDict(sorted(F1.items(), key=lambda t: t[1],reverse=True))

    # for key,value in F1.items():
    #     result_list.append(key+' f1pI,'+str(value))
    # result_list.append("---,---")

    intent_macro_precision = sum(precision.values()) / len(precision)
    intent_macro_recall = sum(recall.values()) / len(recall)
    intent_macro_f1 = sum(F1.values()) / len(F1)
    result_list.append('intent macro precision,'+str(100 * intent_macro_precision))
    result_list.append('intent macro recall,'+str(100 * intent_macro_recall))
    result_list.append('intent macro f1,'+str(100 * intent_macro_f1))
    result_list.append("---,---")
  
    acc_per_intent = os.path.join(output_dir, 'evaluation_per_intent.json')
    with open(acc_per_intent,'w') as file:
        per_slot_dict={}
        per_slot_dict["precision"]=precision
        per_slot_dict["recall"]=recall
        per_slot_dict["F1"]=F1
        json.dump(per_slot_dict,file, indent=4, ensure_ascii=False)

    precision, recall, F1 = calculateF1perSlotCLS(predict_golden['slot'])
    precision = OrderedDict(sorted(precision.items(), key=lambda t: t[1],reverse=True))
    recall = OrderedDict(sorted(recall.items(), key=lambda t: t[1],reverse=True))
    F1 = OrderedDict(sorted(F1.items(), key=lambda t: t[1],reverse=True))

    # for key,value in F1.items():
    #     result_list.append(key+' f1pS,'+str(value))
    # result_list.append("---,---")

    slot_macro_precision = sum(precision.values()) / len(precision)
    slot_macro_recall = sum(recall.values()) / len(recall)
    slot_macro_f1 = sum(F1.values()) / len(F1)
    result_list.append('slot macro precision,'+str(100 * slot_macro_precision))
    result_list.append('slot macro recall,'+str(100 * slot_macro_recall))
    result_list.append('slot macro f1,'+str(100 * slot_macro_f1))
    result_list.append("---,---")

    precision, recall, F1 = calculateF1perSlot(predict_golden['tag'])
    precision = OrderedDict(sorted(precision.items(), key=lambda t: t[1],reverse=True))
    recall = OrderedDict(sorted(recall.items(), key=lambda t: t[1],reverse=True))
    F1 = OrderedDict(sorted(F1.items(), key=lambda t: t[1],reverse=True))

    # for key,value in F1.items():
    #     result_list.append(key+' f1pS,'+str(value))
    # result_list.append("---,---")

    slot_macro_precision = sum(precision.values()) / len(precision)
    slot_macro_recall = sum(recall.values()) / len(recall)
    slot_macro_f1 = sum(F1.values()) / len(F1)
    result_list.append('tag macro precision,'+str(100 * slot_macro_precision))
    result_list.append('tag macro recall,'+str(100 * slot_macro_recall))
    result_list.append('tag macro f1,'+str(100 * slot_macro_f1))
    result_list.append("---,---")
  
    # result logging
    acc_per_slot = os.path.join(output_dir, 'evaluation_per_slot.json')
    with open(acc_per_slot,'w') as file:
        per_slot_dict={}
        per_slot_dict["precision"]=precision
        per_slot_dict["recall"]=recall
        per_slot_dict["F1"]=F1
        json.dump(per_slot_dict,file, indent=4, ensure_ascii=False)

    output_intent = os.path.join(output_dir, 'output_intent.json')
    output_slot = os.path.join(output_dir, 'output_slot.json')
    json.dump(predict_golden['intent'], open(output_intent, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(predict_golden['slot'], open(output_slot, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    output_result = os.path.join(output_dir, 'output_result.csv')

    with open(output_result,'w') as file:
        for result in result_list:
            file.write(result+"\n")

    output_file = os.path.join(output_dir, 'output.json')
    json.dump(predict_golden['tag'], open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
