import os
import sys
import json


def readin(file: str):
    
    data = []
    decoder = json.JSONDecoder()
    with open(file, 'r') as f:
        for line in f.readlines():
            data.append(decoder.decode(line))

    return data

def get_entities(ners: list, length: int, offset: int):

    tags = ['O' for i in range(length)]
    entities = []

    for e in ners:
        start = e[0] - offset
        end = e[1] - offset
        tag = e[2]
        if end == start:
            tags[start] = 'U-' + tag
        else:
            tags[start] = 'B-' + tag
            tags[end] = 'L-' + tag 
                
            for j in range(start+1,end):
                tags[j] = 'I-' + tag 

    return tags

def get_relations(relations: list, ners: list):

    rels = []

    entities = [(e[0], e[1]) for e in ners]
    entities.sort(key=lambda x:x[0])


    for rel in relations:
        head = (rel[0], rel[1])
        tail = (rel[2], rel[3])
        assert head in entities and tail in entities
        rels.append({"head": entities.index(head), 
                    "tail": entities.index(tail),
                    "type": rel[4]})

    return rels

def dygiepp2tabcnn(src_path: str, tgt_path: str):

    files = [f for f in os.listdir(src_path) if f.endswith(".json")]
    for f in files:
        new_data = []
        data = readin(os.path.join(src_path, f))

        for d in data: # every document
            offset = 0
            for i in range(len(d["sentences"])):

                curr_data = {"tokens": d["sentences"][i], 
                            "tags": [], 
                            "relations": [], 
                            "orig_id": f"{d['doc_key']}_{i}"}
                
                curr_data["tags"] = get_entities(d["ner"][i], len(curr_data["tokens"]), offset)
                curr_data["relations"] = get_relations(d["relations"][i], d["ner"][i])
                
                assert len(curr_data["relations"]) == len(d["relations"][i])
                
                offset += len(curr_data["tokens"]) 
                new_data.append(curr_data)
        
        with open(os.path.join(tgt_path, f), "w") as w:
            json.dump(new_data, w)




    return


if __name__ == '__main__':
    if "-h" in sys.argv[1]:
        print("python3 dygiepp2tabcnn.py SRC_DIR TGT_DIR")
    else:
        dygiepp2tabcnn(sys.argv[1], sys.argv[2])
