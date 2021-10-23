import json
import os
import zipfile
import sys
from collections import Counter,OrderedDict


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append([intent, slot, value])
    return triples


def preprocess(mode):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, 'raw')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_da = []
    all_intent = []
    all_slots = []
    all_tag = []
    context_size = 3
    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            context = []
            for is_sys, turn in enumerate(sess['log']):
                if mode == 'usr' and is_sys % 2 == 1:
                    context.append(turn['text'])
                    continue
                elif mode == 'sys' and is_sys % 2 == 0:
                    context.append(turn['text'])
                    continue
                tokens = turn["text"].split()
                dialog_act = {}
                for dacts in turn["span_info"]:
                    if dacts[0] not in dialog_act:
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4] + 1])])

                spans = turn["span_info"]
                tags = []
                slots = []
                for i, _ in enumerate(tokens):
                    for span in spans:
                        slots.append(span[0] + "+" + span[1])
                        if i == span[3]:
                            tags.append("B-" + span[0] + "+" + span[1])
                            # slots.append(span[0] + "+" + span[1])
                            break
                        if span[3] < i <= span[4]:
                            tags.append("I-" + span[0] + "+" + span[1])
                            # slots.append(span[0] + "+" + span[1])
                            break
                    else:
                        tags.append("O")

                intents = []
                for dacts in turn["dialog_act"]:
                    for dact in turn["dialog_act"][dacts]:
                        if dacts not in dialog_act or dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                            if dact[1] in ["none", "?", "yes", "no", "do nt care", "do n't care", "dontcare"]:
                                # intents.append(dacts + "+" + dact[0] + "*" + dact[1])
                                intents.append(dacts)
                processed_data[key].append([tokens, tags, intents, da2triples(turn["dialog_act"]), context[-context_size:]])
                all_da += [da for da in turn['dialog_act']]
                all_intent += intents
                all_slots +=slots
                all_tag += tags

                context.append(turn['text'])

        sorted_da_dict = OrderedDict(sorted(dict(Counter(all_da)).items(), key=lambda t: t[1],reverse=True))
        sorted_intent_dict = OrderedDict(sorted(dict(Counter(all_intent)).items(), key=lambda t: t[1],reverse=True))
        sorted_slot_dict = OrderedDict(sorted(dict(Counter(all_slots)).items(), key=lambda t: t[1],reverse=True))
        all_intent_new=[]
        all_slot_new=[]
        all_tags_new=[]

        new_sample_set=[]
        for i in range(len(processed_data[key])):
            item = processed_data[key][i]
            item_intent_old=item[2]
            item_intent_new=[]
            # item_slot_old=item[3]
            # item_slot_new=[]

            for intent in item_intent_old:
                if(sorted_intent_dict[intent]>20):
                    item_intent_new.append(intent)
                    all_intent_new.append(intent)

            # for slot in item_slot_old:
            #     slot_str=slot[0] + "+" + slot[1]
            #     if(slot_str in sorted_slot_dict and sorted_slot_dict[slot_str]>10):
            #         item_slot_new.append(slot)
            #         all_slot_new.append(slot_str)

            if(len(item_intent_old)==len(item_intent_new)):
                new_sample_set.append(item)
        processed_data[key]=new_sample_set

        all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
        all_intent = [x[0] for x in dict(Counter(all_intent_new)).items() if x[1]]
        all_slots = [x[0] for x in dict(Counter(all_slots)).items() if x[1]]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]



        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key], open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w'), indent=2)

    # print('dialog act num:', len(all_da))
    print('Intent num:', len(all_intent))
    print('Slot num:', len(all_slots))
    print('Tag num:', len(all_tag))
    json.dump(all_da, open(os.path.join(processed_data_dir, 'all_act.json'), 'w'), indent=2)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w'), indent=2)
    json.dump(all_slots, open(os.path.join(processed_data_dir, 'slot_vocab.json'), 'w'), indent=2)
    json.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w'), indent=2)

    json.dump(sorted_da_dict, open(os.path.join(processed_data_dir, 'act_freq.json'), 'w'), indent=2)
    json.dump(sorted_intent_dict, open(os.path.join(processed_data_dir, 'intent_freq.json'), 'w'), indent=2)
    json.dump(sorted_slot_dict, open(os.path.join(processed_data_dir, 'slot_freq.json'), 'w'), indent=2)


if __name__ == '__main__':
    mode = sys.argv[1]
    preprocess(mode)
