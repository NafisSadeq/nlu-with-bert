import argparse
import os
import json
import random
import numpy as np
import torch
from dataloader import Dataloader
from jointBERT import JointBERT
from collections import OrderedDict
from postprocess import is_slot_da, calculateF1, calculateF1perIntent, calculateF1perSlot, recover_intent, recover_slot

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path',
                    help='path to config file')


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']

    set_seed(config['seed'])
       
    print("Data Dir:",data_dir)

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    slot_vocab = json.load(open(os.path.join(data_dir, 'slot_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
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

    model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
    model.to(DEVICE)
    model.eval()

    batch_size = config['model']['batch_size']

    data_key = 'test'
    predict_golden = {'intent': [], 'slot': []}
    slot_loss, intent_loss = 0, 0
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=data_key):
        pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None

        with torch.no_grad():
            slot_logits, intent_logits, batch_slot_loss, batch_intent_loss = model.forward(word_seq_tensor,
                                                                                           word_mask_tensor,
                                                                                           tag_seq_tensor,
                                                                                           tag_mask_tensor,
                                                                                           intent_tensor,
                                                                                           context_seq_tensor,
                                                                                           context_mask_tensor)
        slot_loss += batch_slot_loss.item() * real_batch_size
        intent_loss += batch_intent_loss.item() * real_batch_size
        for j in range(real_batch_size):
            slot_predicts = recover_slot(dataloader, intent_logits[j], slot_logits[j], tag_mask_tensor[j],
                                      ori_batch[j][0], ori_batch[j][-4])
            intent_predicts = recover_intent(dataloader, intent_logits[j])
            slot_labels = ori_batch[j][3]
            intent_labels = ori_batch[j][2]

            predict_golden['slot'].append({
                'predict': [x for x in slot_predicts if is_slot_da(x)],
                'golden': [x for x in slot_labels if is_slot_da(x)]
            })

            predict_golden['intent'].append({
                'predict': intent_predicts,
                'golden': intent_labels
            })
        print('[%d|%d] samples' % (len(predict_golden['slot']), len(dataloader.data[data_key])))

    total = len(dataloader.data[data_key])
    slot_loss /= total
    intent_loss /= total
    print('%d samples %s' % (total, data_key))
    print('\t slot loss:', slot_loss)
    print('\t intent loss:', intent_loss)

    for x in ['intent', 'slot']:
        precision, recall, F1 = calculateF1(predict_golden[x],x)
        print('-' * 20 + x + '-' * 20)
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

    precision, recall, F1 = calculateF1perIntent(predict_golden['intent'])
    precision = OrderedDict(sorted(precision.items(), key=lambda t: t[1],reverse=True))
    recall = OrderedDict(sorted(recall.items(), key=lambda t: t[1],reverse=True))
    F1 = OrderedDict(sorted(F1.items(), key=lambda t: t[1],reverse=True))
    # print('-' * 20 + "Intent" + '-' * 20)
    # print('\t Precision: ',json.dumps(precision, indent=4))
    # print('\t Recall: ',json.dumps(recall, indent=4))
    # print('\t F1: ',json.dumps(F1, indent=4))

    acc_per_intent = os.path.join(output_dir, 'evaluation_per_intent.json')
    with open(acc_per_intent,'w') as file:
        json.dump(precision,file, indent=4, ensure_ascii=False)
        json.dump(recall,file, indent=4, ensure_ascii=False)
        json.dump(F1,file, indent=4, ensure_ascii=False)

    precision, recall, F1 = calculateF1perSlot(predict_golden['slot'])
    precision = OrderedDict(sorted(precision.items(), key=lambda t: t[1],reverse=True))
    recall = OrderedDict(sorted(recall.items(), key=lambda t: t[1],reverse=True))
    F1 = OrderedDict(sorted(F1.items(), key=lambda t: t[1],reverse=True))
    # print('-' * 20 + "Slot" + '-' * 20)
    # print('\t Precision: ',json.dumps(precision, indent=4))
    # print('\t Recall: ',json.dumps(recall, indent=4))
    # print('\t F1: ',json.dumps(F1, indent=4))

    acc_per_slot = os.path.join(output_dir, 'evaluation_per_slot.json')
    with open(acc_per_slot,'w') as file:
        json.dump(precision,file, indent=4, ensure_ascii=False)
        json.dump(recall,file, indent=4, ensure_ascii=False)
        json.dump(F1,file, indent=4, ensure_ascii=False)

    output_file = os.path.join(output_dir, 'output.json')
    json.dump(predict_golden['slot'], open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
