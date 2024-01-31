# 复现


```bash
pip install -r requirements.txt -U


# KB-BINDER
python few_shot_kbqa.py --shot_num 4 --temperature 0.3 \
    --engine text-davinci-003 \
    --fb_roles_path data/fb_roles \
    --surface_map_path data/surface_map_file_freebase_complete_all_mention \
    --train_data_path data/webqsp_0107.train.json \
    --eva_data_path data/webqsp_0107.test.json

# KB-BINDER-R
python few_shot_kbqa.py --shot_num 4 --temperature 0.3 \
    --engine text-davinci-003 --retrieval \
    --fb_roles_path data/fb_roles \
    --surface_map_path data/surface_map_file_freebase_complete_all_mention \
    --train_data_path data/webqsp_0107.train.json \
    --eva_data_path data/webqsp_0107.test.json
```

# my

```bash

export JAVA_HOME=/home/${USER}/opt/jdk-17.0.9
python few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
    --engine gpt-4-1106-preview \
    --fb_roles_path data/fb_roles \
    --surface_map_path data/surface_map_file_freebase_complete_all_mention \
    --train_data_path smalldata/WebQSP_train.json \
    --eva_data_path smalldata/WebQSP_test.json

# export OPENAI_API_KEY=sk-123
# export JAVA_HOME=/home/${USER}/opt/jdk-17.0.9
# python few_shot_kbqa.py --shot_num 4 --temperature 0.3 --retrieval \
#     --engine gpt-4-1106-preview \
#     --fb_roles_path data/fb_roles \
#     --surface_map_path data/surface_map_file_freebase_complete_all_mention \
#     --train_data_path smalldata/WebQSP_train.json \
#     --eva_data_path smalldata/WebQSP_test.json

# metaqa
export OPENAI_API_KEY=sk-123
python metaqa_src/metaqa_1hop.py &
python metaqa_src/metaqa_2hop.py &
python metaqa_src/metaqa_3hop.py &
```

ls save/metaqa/*-hop/KB-BINDER-gpt-4-1106-preview/*.json | wc -l

ls save/metaqa/*-hop/KB-BINDER-R-gpt-4-1106-preview/*.json | wc -l

ls save/webqsp/KB-BINDER-gpt-4-1106-preview/*.json | wc -l
ls save/webqsp/KB-BINDER-R-gpt-4-1106-preview/*.json | wc -l