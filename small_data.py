import os
import json
from glob import glob


def make_small_data(dataset):
    assert dataset in ["WebQSP","MetaQA"]
    paths = glob(f"../LLM_KGQA/save-anno-clean/{dataset.lower()}/**/*.json")
    train_data = [json.load(open(path)) for path in paths]
    train_ids = [d["id"].replace(".P0","") for d in train_data]
    train_ids = set(train_ids)
    print("len(train_ids)", len(train_ids))
    
    this_data_train = json.load(open(f"data/webqsp_0107.train.json"))
    this_data_train = [d for d in this_data_train if d["qid"] in train_ids]
    print("len(this_data_train)", len(this_data_train))
    
    # save to save_dir_fid_smalldata/{dataset}_{split}_SPQA.json
    save_dir = "smalldata"
    os.makedirs(save_dir, exist_ok=True)

    out = f"{save_dir}/{dataset}_train.json"
    print(f"Saving to {out}")
    json.dump(this_data_train, open(out, "w"), indent=4, ensure_ascii=False)

    # test
    paths = glob(f"../LLM_KGQA/data/{dataset.lower()}/test/*.json")
    test_data = []
    for path in paths:
        with open(path) as f:
            test_data += json.load(f)
    test_ids = [d["id"].replace(".P0","") for d in test_data]
    test_ids = set(test_ids)
    print("len(test_ids)", len(test_ids))

    this_data_test = json.load(open(f"data/webqsp_0107.test.json"))
    this_data_test = [d for d in this_data_test if d["qid"] in test_ids]
    print("len(this_data_test)", len(this_data_test))
    
    out = f"{save_dir}/{dataset}_test.json"
    print(f"Saving to {out}")
    json.dump(this_data_test, open(out, "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    """
    python small_data.py
    """
    make_small_data("WebQSP")
    # make_small_data("CWQ")

    # tar cvzf save_dir_fid_smalldata.tar.gz save_dir_fid_smalldata
