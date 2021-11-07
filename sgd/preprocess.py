import json
import os
import zipfile
import sys
from collections import Counter,OrderedDict
from tqdm import tqdm

def transformIntentName(word):
    index=0
    for i in range(1,len(word)):
        c=word[i]
        # print(i,c)
        if(c.isupper()):
            index=i
            return word[i:]+"-"+word[:i]
    return word[index:]+"+"+word[:index]

def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

def read_json(filepath, filename):
    path = os.path.join(filepath, filename)
    return json.load(open(path, "r")) 

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
        for d in tqdm(data[key]):
            context = []
            for turn in d['turns']:

                is_sys = (turn['speaker'] == 'SYSTEM')
                if mode == 'usr' and is_sys:
                    context.append(turn['utterance'])
                    continue
                elif mode == 'sys' and is_sys:
                    context.append(turn['utterance'])
                    continue
                tokens = turn['utterance'].split()
                tokens_map = {}
                start = 0
                for i, t in enumerate(tokens):
                    tokens_map[(i, t)] = (start, start+len(t))
                    start += len(t) + 1

                tags = []
                slots = []

                for i, _ in enumerate(tokens):
                    tags.append("O")

                intents = []
                actions = []
                for frame in turn["frames"]:
                    if "state" in frame:
                        intent_name=transformIntentName(frame["state"]["active_intent"])
                        intents.append(intent_name)
                        if "actions" in frame:
                            assert "state" in frame
                            slots.extend([[intent_name, i['slot'], i['values'][0] if len(i['values']) else '?'] for i in frame["actions"]])
                            actions.extend([i['act'] for i in frame["actions"]])
                        for s in frame['slots']:
                            for i, t in enumerate(tokens):
                                if s['start'] == tokens_map[(i, t)][0]:
                                    assert "state" in frame
                                    tags[i] = f"B-{intent_name}+{s['slot']}"
                                elif tokens_map[(i, t)][0] > s['start'] and tokens_map[(i, t)][0] <= s['exclusive_end']:
                                    assert "state" in frame
                                    tags[i] = f"I-{intent_name}+{s['slot']}"                              

                processed_data[key].append([tokens, tags, intents, slots, context[-context_size:]])

                all_da += actions
                all_intent += intents
                all_slots += [s1+'+'+s2 for s1, s2, _ in slots]
                all_tag += tags

                context.append(turn['utterance'])

        sorted_da_dict = OrderedDict(sorted(dict(Counter(all_da)).items(), key=lambda t: t[1],reverse=True))
        sorted_intent_dict = OrderedDict(sorted(dict(Counter(all_intent)).items(), key=lambda t: t[1],reverse=True))
        sorted_slot_dict = OrderedDict(sorted(dict(Counter(all_slots)).items(), key=lambda t: t[1],reverse=True))

        all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
        all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
        all_slots = [x[0] for x in dict(Counter(all_slots)).items() if x[1]]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]

        if (key=='test'):
            for i in range(len(processed_data[key])):
                item = processed_data[key][i]
                item_intent_old=item[2]
                item_intent_new=[]

                for intent in item_intent_old:
                    if(sorted_intent_dict[intent]>32):
                        item_intent_new.append(intent)

                processed_data[key][i][2]=item_intent_new

        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key], open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w'), indent=2)

    print('dialog act num:', len(all_da))
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
