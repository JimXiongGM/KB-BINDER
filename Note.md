# 复现

KB-BINDER
```
python few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
    --engine gpt-3.5-turbo-16k \
    --fb_roles_path data/fb_roles --surface_map_path [your surface map file path] \
    --train_data_path data/webqsp_0107.train.json --eva_data_path data/webqsp_0107.test.json
```

