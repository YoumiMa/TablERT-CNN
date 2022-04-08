import json
import copy
import os
import sys


def intersects(tup1, tup2):
    return tup2[0][0] <= tup1[0][0] < tup2[0][1] or tup2[0][0] <= tup1[0][1] - 1 < tup2[0][1]


def get_entities(curr_ents: list, curr_tags: list, entities: list):

    for ent_i in curr_ents:
        start = entities[ent_i][0][0]
        end = entities[ent_i][0][1] - 1
        tag = entities[ent_i][1]
        if end == start:
            curr_tags[start] = 'U-' + tag
        else:
            curr_tags[start] = 'B-' + tag
            curr_tags[end] = 'L-' + tag 
                
            for j in range(start+1,end):
                curr_tags[j] = 'I-' + tag 

    return curr_tags

def get_relations(relations: list, entities: list, curr_ents: list):

    curr_rels = []        

    for rel in relations:
        curr_ents.sort(key=lambda x: entities[x][0])
        if rel["head"] in curr_ents and rel["tail"] in curr_ents:
            curr_rels.append({"head": curr_ents.index(rel["head"]), "tail": curr_ents.index(rel["tail"]), "type": rel['type']})    
   
    return curr_rels

def spert2tabcnn(src_path: str, tgt_path: str):

    files = [f for f in os.listdir(src_path) if f.endswith(".json") and "types" not in f]
    
    for file in files:
        with open(os.path.join(src_path, file), 'r') as f:
            data = json.load(f)

        new_data = []

        for d in data:
            ents = d["entities"]
            entities = [((ent["start"], ent["end"]), ent["type"]) for ent in ents]
            entities_dict = {"non-overlapped": [i for i in range(len(entities))], "overlapped": []}
            
            # check overlap between entity spans.
            for i in range(len(entities)):
                for j in range(i+1,len(entities)):
                    if intersects(entities[i], entities[j]) or intersects(entities[j], entities[i]):
                        entities_dict["overlapped"].append((i,j))
                        if i in entities_dict["non-overlapped"]:
                            entities_dict["non-overlapped"].remove(i) 
                        if j in entities_dict["non-overlapped"]:
                            entities_dict["non-overlapped"].remove(j)

            curr_ents = copy.deepcopy(entities_dict["non-overlapped"])
            curr_tags = ["O" for i in range(len(d['tokens']))]

            # bilou entity tags.
            curr_tags = get_entities(curr_ents, curr_tags, entities)  

            # relations.     
            curr_rels= get_relations(d["relations"], entities, curr_ents)

            new_data.append({"tokens": d["tokens"], "tags": curr_tags , "relations": curr_rels, "orig_id": d["orig_id"]})

        with open(os.path.join(tgt_path, file), 'w') as w:
            json.dump(new_data, w)
    return



if __name__ == '__main__':
    if "-h" in sys.argv[1]:
        print("python3 spert2tabcnn.py SRC_DIR TGT_DIR")
    else:
        spert2tabcnn(sys.argv[1], sys.argv[2])
