import os

os.environ["https_proxy"] = "http://127.0.0.1:7893"
os.environ["http_proxy"] = "http://127.0.0.1:7893"

import argparse
import itertools
import json

import random
import re
from collections import Counter
from time import sleep
from tqdm import tqdm

import openai
import spacy
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from rank_bm25 import BM25Okapi
from loguru import logger
from sparql_exe import execute_query, get_2hop_relations, get_types, lisp_to_sparql
from utils import process_file, process_file_node

DEFAULT_KEY = os.environ.get("OPENAI_API_KEY", None)
assert DEFAULT_KEY is not None, "OPENAI_API_KEY is None"
client = openai.OpenAI(api_key=DEFAULT_KEY)


def select_shot_prompt_train(train_data_in, shot_number):
    """
    本函数对问题进行了简单的分类，分为了两类，一类是包含了比较关系的问题，一类是不包含比较关系的问题。
    一些观察
    - webqsp数据，compare类型的数据例如：
    ['where did galileo go to school', 'who does joakim noah play for', 'what did doctor kevorkian do', 'who is the president of israel 2012',
        除了最后一个，前面的并没有比较关系。
    """
    train_data_in = [d for d in train_data_in if d["s_expression"]]
    random.shuffle(train_data_in)
    compare_list = ["le", "ge", "gt", "lt", "ARGMIN", "ARGMAX"]
    if shot_number == 1:
        selected_quest_compose = [train_data_in[0]["question"]]
        selected_quest_compare = [train_data_in[0]["question"]]
        selected_quest = [train_data_in[0]["question"]]
    else:
        selected_quest_compose = []
        selected_quest_compare = []
        each_type_num = shot_number // 2
        for data in train_data_in:
            if any([x in data["s_expression"] for x in compare_list]):
                selected_quest_compare.append(data["question"])
                if len(selected_quest_compare) == each_type_num:
                    break
        for data in train_data_in:
            if not any([x in data["s_expression"] for x in compare_list]):
                selected_quest_compose.append(data["question"])
                if len(selected_quest_compose) == each_type_num:
                    break
        mix_type_num = each_type_num // 3
        selected_quest = (
            selected_quest_compose[:mix_type_num]
            + selected_quest_compare[:mix_type_num]
        )
    logger.info("selected_quest_compose: {}".format(selected_quest_compose))
    logger.info("selected_quest_compare: {}".format(selected_quest_compare))
    logger.info("selected_quest: {}".format(selected_quest))
    return selected_quest_compose, selected_quest_compare, selected_quest


def sub_mid_to_fn(question, string, question_to_mid_dict):
    seg_list = string.split()
    mid_to_start_idx_dict = {}
    for seg in seg_list:
        if seg.startswith("m.") or seg.startswith("g."):
            mid = seg.strip(")(")
            start_index = string.index(mid)
            mid_to_start_idx_dict[mid] = start_index
    if len(mid_to_start_idx_dict) == 0:
        return string
    start_index = 0
    new_string = ""
    for key in mid_to_start_idx_dict:
        b_idx = mid_to_start_idx_dict[key]
        e_idx = b_idx + len(key)
        new_string = (
            new_string + string[start_index:b_idx] + question_to_mid_dict[question][key]
        )
        start_index = e_idx
    new_string = new_string + string[start_index:]
    return new_string


def type_generator(question, prompt_type, LLM_engine):
    """
    这个函数先生成问题的type？
    """
    sleep(1)
    prompt = prompt_type
    prompt = prompt + " Question: " + question + "\nType of the question: "
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": prompt},
    ]
    got_result = False
    while got_result != True:
        try:
            response = client.chat.completions.create(
                model=LLM_engine,
                messages=messages,
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["Question: "],
            )
            response = json.loads(response.json())
            gene_type = response["choices"][0]["message"]["content"].strip()
            return gene_type, response["usage"]
        except Exception as e:
            print("error in type generation", e)
            sleep(3)


def ep_generator(
    question,
    selected_examples,
    temp,
    que_to_s_dict_train,
    question_to_mid_dict,
    LLM_engine,
    retrieval=False,
    corpus=None,
    nlp_model=None,
    bm25_train_full=None,
    retrieve_number=100,
):
    if retrieval:
        tokenized_query = nlp_model(question)
        tokenized_query = [token.lemma_ for token in tokenized_query]
        top_ques = bm25_train_full.get_top_n(tokenized_query, corpus, n=retrieve_number)
        doc_scores = bm25_train_full.get_scores(tokenized_query)
        top_score = max(doc_scores)
        logger.info("top_score: {}".format(top_score))
        logger.info("top related questions: {}".format(top_ques))
        selected_examples = top_ques
    
    instruction = "Please write the s-expression for the given question based on the format provided.\n\n"
    prompt = ""
    for que in selected_examples:
        if not que_to_s_dict_train[que]:
            continue
        prompt = (
            prompt
            + "Question: "
            + que
            + "\n"
            + "Logical Form: "
            + sub_mid_to_fn(que, que_to_s_dict_train[que], question_to_mid_dict)
            + "\n"
        )
    prompt = instruction + prompt + "Question: " + question + "\n" # + "Logical Form: "
    
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant. Follow the format to complete the task.",
        },
        {"role": "user", "content": prompt},
    ]

    with open(f".temp-out-R{retrieval}.txt", "a") as f:
        f.write(prompt + "\nEND!")

    got_result = False
    while got_result != True:
        try:
            # 只有这里保存了多个生成结果
            response = client.chat.completions.create(
                model=LLM_engine,
                messages=messages,
                temperature=temp,
                top_p=1,
                n=7,
                stop=["Question: ", "\n"],
                max_tokens=256,
                presence_penalty=0,
                frequency_penalty=0,
            )
            response = json.loads(response.json())
            gene_exp = [
                exp["message"]["content"].strip() for exp in response["choices"]
            ]
            for i in range(len(gene_exp)):
                gene_exp[i] = gene_exp[i].replace("Logical Form: ", "")
            print("gene exp all:", gene_exp)
            return gene_exp, response["usage"]
        except Exception as e:
            print("error in ep generation", e)
            sleep(3)


def convert_to_frame(s_exp):
    phrase_set = [
        "(JOIN",
        "(ARGMIN",
        "(ARGMAX",
        "(R",
        "(le",
        "(lt",
        "(ge",
        "(gt",
        "(COUNT",
        "(AND",
        "(TC",
        "(CONS",
    ]
    seg_list = s_exp.split()
    after_filter_list = []
    for seg in seg_list:
        for phrase in phrase_set:
            if phrase in seg:
                after_filter_list.append(phrase)
        if ")" in seg:
            after_filter_list.append("".join(i for i in seg if i == ")"))
    return "".join(after_filter_list)


def find_friend_name(gene_exp, org_question):
    """
    一个很简单的friend name解析器？
    """
    seg_list = gene_exp.split()
    phrase_set = [
        "(JOIN",
        "(ARGMIN",
        "(ARGMAX",
        "(R",
        "(le",
        "(lt",
        "(ge",
        "(gt",
        "(COUNT",
        "(AND",
    ]
    temp = []
    reg_ents = []
    for i, seg in enumerate(seg_list):
        if not any([ph in seg for ph in phrase_set]):
            if seg.lower() in org_question:
                temp.append(seg.lower())
            if seg.endswith(")"):
                stripped = seg.strip(")")
                stripped_add = stripped + ")"
                if stripped_add.lower() in org_question:
                    temp.append(stripped_add.lower())
                    reg_ents.append(" ".join(temp).lower())
                    temp = []
                elif stripped.lower() in org_question:
                    temp.append(stripped.lower())
                    reg_ents.append(" ".join(temp).lower())
                    temp = []
    if len(temp) != 0:
        reg_ents.append(" ".join(temp))
    return reg_ents


def get_right_mid_set(fn, id_dict, question):
    """
    id_dict: {mid: score} list
    """
    type_to_mid_dict = {}
    type_list = []
    for mid in id_dict:
        types = get_types(mid)
        for cur_type in types:
            if not cur_type.startswith("common.") and not cur_type.startswith("base."):
                if cur_type not in type_to_mid_dict:
                    type_to_mid_dict[cur_type] = {}
                    type_to_mid_dict[cur_type][mid] = id_dict[mid]
                else:
                    type_to_mid_dict[cur_type][mid] = id_dict[mid]
                type_list.append(cur_type)
    tokenized_type_list = [re.split("\.|_", doc) for doc in type_list]
    #     tokenized_question = tokenizer.tokenize(question)
    tokenized_question = question.split()
    bm25 = BM25Okapi(tokenized_type_list)

    # 这里居然把mids的type全部搜集到一起，然后以tokenized question为query，搜集top10的type？？为什么能用分词后的question来搜集type呢？猜测：question中会出现type的名字，例如：'what does jamaican people speak'
    top10_types = bm25.get_top_n(tokenized_question, type_list, n=10)

    selected_types = top10_types[:3]  # 为什么不直接n=3呢？
    selected_mids = []
    for any_type in selected_types:
        # logger.info("any_type: {}".format(any_type))
        # logger.info("type_to_mid_dict[any_type]: {}".format(type_to_mid_dict[any_type]))
        selected_mids += list(type_to_mid_dict[any_type].keys())
    return selected_mids


def from_fn_to_id_set(fn_list, question, name_to_id_dict, bm25_all_fns, all_fns):
    return_mid_list = []
    for fn_org in fn_list:
        drop_dot = fn_org.split()
        drop_dot = [seg.strip(".") for seg in drop_dot]
        drop_dot = " ".join(drop_dot)
        if fn_org.lower() not in question and drop_dot.lower() in question:
            fn_org = drop_dot
        if fn_org.lower() not in name_to_id_dict:
            logger.info("fn_org: {}".format(fn_org.lower()))
            tokenized_query = fn_org.lower().split()
            fn = bm25_all_fns.get_top_n(tokenized_query, all_fns, n=1)[0]
            logger.info("sub fn: {}".format(fn))
        else:
            fn = fn_org
        if fn.lower() in name_to_id_dict:
            id_dict = name_to_id_dict[fn.lower()]

        # name 对应的 mids 数量太多
        if len(id_dict) > 15:
            mids = get_right_mid_set(fn.lower(), id_dict, question)
        else:
            mids = sorted(id_dict.items(), key=lambda x: x[1], reverse=True)
            mids = [mid[0] for mid in mids]
        return_mid_list.append(mids)
    return return_mid_list


def convz_fn_to_mids(gene_exp, found_names, found_mids):
    if len(found_names) == 0:
        return gene_exp
    start_index = 0
    new_string = ""
    for name, mid in zip(found_names, found_mids):
        b_idx = gene_exp.lower().index(name)
        e_idx = b_idx + len(name)
        new_string = new_string + gene_exp[start_index:b_idx] + mid
        start_index = e_idx
    new_string = new_string + gene_exp[start_index:]
    return new_string


def add_reverse(org_exp):
    final_candi = [org_exp]
    total_join = 0
    list_seg = org_exp.split(" ")
    for seg in list_seg:
        if "JOIN" in seg:
            total_join += 1
    for i in range(total_join):
        final_candi = final_candi + add_reverse_index(final_candi, i + 1)
    return final_candi


def add_reverse_index(list_of_e, join_id):
    added_list = []
    list_of_e_copy = list_of_e.copy()
    for exp in list_of_e_copy:
        list_seg = exp.split(" ")
        count = 0
        for i, seg in enumerate(list_seg):
            if "JOIN" in seg and "." in list_seg[i + 1]:
                count += 1
                if count != join_id:
                    continue
                list_seg[i + 1] = "(R " + list_seg[i + 1] + ")"
                added_list.append(" ".join(list_seg))
                break
            if "JOIN" in seg and "(R" in list_seg[i + 1]:
                count += 1
                if count != join_id:
                    continue
                list_seg[i + 1] = ""
                list_seg[i + 2] = list_seg[i + 2][:-1]
                added_list.append(" ".join(" ".join(list_seg).split()))
                break
    return added_list


def bound_to_existed(
    question,
    s_expression,
    found_mids,
    two_hop_rela_dict,
    relationship_to_enti,
    hsearcher,
    rela_corpus,
    relationships,
):
    possible_relationships_can = []
    possible_relationships = []

    logger.info("before 2 hop rela")
    updating_two_hop_rela_dict = two_hop_rela_dict.copy()
    for mid in found_mids:
        if mid in updating_two_hop_rela_dict:
            relas = updating_two_hop_rela_dict[mid]
            possible_relationships_can += list(set(relas[0]))
            possible_relationships_can += list(set(relas[1]))
        else:
            relas = get_2hop_relations(mid)
            updating_two_hop_rela_dict[mid] = relas
            possible_relationships_can += list(set(relas[0]))
            possible_relationships_can += list(set(relas[1]))

    logger.info("after 2 hop rela")
    for rela in possible_relationships_can:
        if (
            not rela.startswith("common")
            and not rela.startswith("base")
            and not rela.startswith("type")
        ):
            possible_relationships.append(rela)
    if not possible_relationships:
        possible_relationships = relationships.copy()
    expression_segment = s_expression.split(" ")

    print("possible_relationships: ", possible_relationships)
    possible_relationships = list(set(possible_relationships))
    relationship_replace_dict = {}
    lemma_tags = {"NNS", "NNPS"}
    for i, seg in enumerate(expression_segment):
        processed_seg = seg.strip(")")
        if (
            "." in seg
            and not seg.startswith("m.")
            and not seg.startswith("g.")
            and not (
                expression_segment[i - 1].endswith("AND")
                or expression_segment[i - 1].endswith("COUNT")
                or expression_segment[i - 1].endswith("MAX")
                or expression_segment[i - 1].endswith("MIN")
            )
            and (not any(ele.isupper() for ele in seg))
        ):
            tokenized_query = re.split("\.|_", processed_seg)
            tokenized_query = " ".join(tokenized_query)
            tokenized_question = question.strip(" ?")
            tokenized_query = tokenized_query + " " + tokenized_question

            # 这里是重点，把谓词和Q拼到一起，然后用hybrid search来搜索relation
            searched_results = hsearcher.search(tokenized_query, k=1000)

            top3_ques = []
            for hit in searched_results:
                if len(top3_ques) > 7:
                    break
                cur_result = json.loads(rela_corpus.doc(str(hit.docid)).raw())
                cur_rela = cur_result["rel_ori"]

                # cur_rela in possible_relationships 是重点，也是做了一个约束，必须在mid的两跳以内
                if (
                    not cur_rela.startswith("base.")
                    and not cur_rela.startswith("common.")
                    and not cur_rela.endswith("_inv.")
                    and len(cur_rela.split(".")) > 2
                    and cur_rela in possible_relationships
                ):
                    top3_ques.append(cur_rela)

            logger.info("top3_ques rela: {}".format(top3_ques))
            relationship_replace_dict[i] = top3_ques[:7]
    if len(relationship_replace_dict) > 5:
        return None, updating_two_hop_rela_dict, None
    elif len(relationship_replace_dict) >= 3:
        for key in relationship_replace_dict:
            relationship_replace_dict[key] = relationship_replace_dict[key][:4]
    combinations = list(relationship_replace_dict.values())
    all_iters = list(itertools.product(*combinations))
    rela_index = list(relationship_replace_dict.keys())
    
    logger.info("all_iters: {}".format(all_iters))
    for iters in all_iters:
        expression_segment_copy = expression_segment.copy()
        possible_entities_set = []
        for i in range(len(iters)):
            suffix = ""
            for k in range(len(expression_segment[rela_index[i]].split(")")) - 1):
                suffix = suffix + ")"
            expression_segment_copy[rela_index[i]] = iters[i] + suffix
            if iters[i] in relationship_to_enti:
                possible_entities_set += relationship_to_enti[iters[i]]
        if not possible_entities_set:
            continue
        enti_replace_dict = {}
        for j, seg in enumerate(expression_segment):
            processed_seg = seg.strip(")")
            if (
                "." in seg
                and not seg.startswith("m.")
                and not seg.startswith("g.")
                and (
                    expression_segment[j - 1].endswith("AND")
                    or expression_segment[j - 1].endswith("COUNT")
                    or expression_segment[j - 1].endswith("MAX")
                    or expression_segment[j - 1].endswith("MIN")
                )
                and (not any(ele.isupper() for ele in seg))
            ):
                tokenized_enti = [
                    re.split("\.|_", doc) for doc in possible_entities_set
                ]
                tokenized_query = re.split("\.|_", processed_seg)
                bm25 = BM25Okapi(tokenized_enti)
                top3_ques = bm25.get_top_n(tokenized_query, possible_entities_set, n=3)
                enti_replace_dict[j] = list(set(top3_ques))
        combinations_enti = list(enti_replace_dict.values())
        all_iters_enti = list(itertools.product(*combinations_enti))
        enti_index = list(enti_replace_dict.keys())
        for iter_ent in all_iters_enti:
            for k in range(len(iter_ent)):
                suffix = ""
                for h in range(len(expression_segment[enti_index[k]].split(")")) - 1):
                    suffix = suffix + ")"
                expression_segment_copy[enti_index[k]] = iter_ent[k] + suffix
            final = " ".join(expression_segment_copy)
            added = add_reverse(final)
            for exp in added:
                try:
                    answer = generate_answer([exp])
                except:
                    answer = None
                if answer is not None:
                    return answer, updating_two_hop_rela_dict, exp
    return None, updating_two_hop_rela_dict, None


def generate_answer(list_exp):
    for exp in list_exp:
        try:
            sparql = lisp_to_sparql(exp)
        except:
            continue
        try:
            re = execute_query(sparql)
        except:
            continue
        if re:
            if re[0].isnumeric():
                if re[0] == "0":
                    continue
                else:
                    return re
            else:
                return re
    return None


def number_of_join(exp):
    count = 0
    seg_list = exp.split()
    for seg in seg_list:
        if "JOIN" in seg:
            count += 1
    return count


def process_file_codex_output(filename_before, filename_after):
    codex_eps_dict_before = json.load(open(filename_before, "r"), strict=False)
    codex_eps_dict_after = json.load(open(filename_after, "r"), strict=False)
    for key in codex_eps_dict_after:
        codex_eps_dict_before[key] = codex_eps_dict_after[key]
    return codex_eps_dict_before


def all_combiner_evaluation(
    item,
    selected_quest_compare,
    selected_quest_compose,
    selected_quest,
    prompt_type,
    hsearcher,
    rela_corpus,
    relationships,
    temp,
    que_to_s_dict_train,
    question_to_mid_dict,
    LLM_engine,
    name_to_id_dict,
    bm25_all_fns,
    all_fns,
    relationship_to_enti,
    retrieval=False,
    corpus=None,
    nlp_model=None,
    bm25_train_full=None,
    retrieve_number=100,
):
    # correct = [0] * 6
    # total = [0] * 6
    # no_ans = [0] * 6
    # for idx, item in enumerate(data_batch):
    if retrieval:
        out = f"save/webqsp/KB-BINDER-R-{LLM_engine}"
    else:
        out = f"save/webqsp/KB-BINDER-{LLM_engine}"
    os.makedirs(out, exist_ok=True)
    # if exisited, skip
    if os.path.exists(f"{out}/{item['id']}.json"):
        return
    # logger.info(f"====={idx}/{len(data_batch)}=====")
    logger.info("item[id]: {}".format(item["id"]))
    logger.info("item[question]: {}".format(item["question"]))
    logger.info("item[exp]: {}".format(item["s_expression"]))
    label = []
    for ans in item["answer"]:
        label.append(ans["answer_argument"])
    if not retrieval:
        gene_type, gene_type_usage = type_generator(
            item["question"], prompt_type, LLM_engine
        )
        logger.info("gene_type: {}".format(gene_type))
    else:
        gene_type = None

    if gene_type == "Comparison":
        gene_exps, gen_exps_usage = ep_generator(
            item["question"],
            list(set(selected_quest_compare) | set(selected_quest)),
            temp,
            que_to_s_dict_train,
            question_to_mid_dict,
            LLM_engine,
            retrieval=retrieval,
            corpus=corpus,
            nlp_model=nlp_model,
            bm25_train_full=bm25_train_full,
            retrieve_number=retrieve_number,
        )
    else:
        gene_exps, gen_exps_usage = ep_generator(
            item["question"],
            list(set(selected_quest_compose) | set(selected_quest)),
            temp,
            que_to_s_dict_train,
            question_to_mid_dict,
            LLM_engine,
            retrieval=retrieval,
            corpus=corpus,
            nlp_model=nlp_model,
            bm25_train_full=bm25_train_full,
            retrieve_number=retrieve_number,
        )
    two_hop_rela_dict = {}
    answer_candi = []
    removed_none_candi = []
    answer_to_grounded_dict = {}
    logger.info("gene_exps: {}".format(gene_exps))
    scouts = gene_exps[:6]
    for idx, gene_exp in enumerate(scouts):
        try:
            logger.info(f"{idx}/{len(scouts)} gene_exp: {gene_exp}")
            join_num = number_of_join(gene_exp)
            if join_num > 5:
                continue
            if join_num > 3:
                top_mid = 5
            else:
                top_mid = 15
            # 从生成的s-expr中解析出mention。居然也不加一个括号什么的
            found_names = find_friend_name(gene_exp, item["question"])
            found_mids = from_fn_to_id_set(
                found_names,
                item["question"],
                name_to_id_dict,
                bm25_all_fns,
                all_fns,
            )
            found_mids = [mids[:top_mid] for mids in found_mids]
            mid_combinations = list(itertools.product(*found_mids))
            # logger.info("all_iters: {}".format(mid_combinations))
            for mid_iters in mid_combinations:
                # logger.info("mid_iters: {}".format(mid_iters))

                # 将s-expr中的mention替换为mid
                replaced_exp = convz_fn_to_mids(gene_exp, found_names, mid_iters)

                answer, two_hop_rela_dict, bounded_exp = bound_to_existed(
                    item["question"],
                    replaced_exp,
                    mid_iters,
                    two_hop_rela_dict,
                    relationship_to_enti,
                    hsearcher,
                    rela_corpus,
                    relationships,
                )
                answer_candi.append(answer)
                if answer is not None:
                    answer_to_grounded_dict[tuple(answer)] = bounded_exp
            for ans in answer_candi:
                if ans != None:
                    removed_none_candi.append(ans)
            if not removed_none_candi:
                answer = None
            else:
                count_dict = Counter([tuple(candi) for candi in removed_none_candi])
                logger.info("count_dict: {}".format(count_dict))
                answer = max(count_dict, key=count_dict.get)
        except:
            if not removed_none_candi:
                answer = None
            else:
                count_dict = Counter([tuple(candi) for candi in removed_none_candi])
                logger.info("count_dict: {}".format(count_dict))
                answer = max(count_dict, key=count_dict.get)

        # 这里还是在6个expr循环中 使用idx来记录每一个expr的结果
        # 6个expr中只需要有一个正确就行？？exact match的定义真是golden中有一个在predic中就算正确，这里放松为6个expr中有一个正确就行？
        answer_to_grounded_dict[None] = ""
        # logger.info("predicted_answer: {}".format(answer))
        # logger.info("label: {}".format(label))
        # if answer is None:
        #     no_ans[idx] += 1
        # elif set(answer) == set(label):
        #     correct[idx] += 1
        # total[idx] += 1
        # em_score = correct[idx] / total[idx]
        # logger.info(
        #     "================================================================"
        # )
        # logger.info("consistent candidates number: {}".format(idx + 1))
        # logger.info("em_score: {}".format(em_score))
        # logger.info("correct: {}".format(correct[idx]))
        # logger.info("total: {}".format(total[idx]))
        # logger.info("no_ans: {}".format(no_ans[idx]))
        # logger.info(" ")
        # logger.info(
        #     "================================================================"
        # )

    # save a out dict for each item
    item["model_name"] = LLM_engine
    item["raw_out"] = scouts
    item["removed_none_candi"] = removed_none_candi
    # item["em_score"] = em_score
    item["gene_type_usage"] = gene_type_usage
    item["gen_exps_usage"] = gen_exps_usage

    # save to "save/webqsp/KB-BINDER-{model_name}/{id}.json"
    logger.info("saving to {}".format(f"{out}/{item['id']}.json"))
    with open(f"{out}/{item['id']}.json", "w") as f:
        json.dump(item, f, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--shot_num",
        type=int,
        metavar="N",
        default=40,
        help="the number of shots used in in-context demo",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="N",
        default=0.3,
        help="the temperature of LLM",
    )
    parser.add_argument(
        "--engine",
        type=str,
        metavar="N",
        default="code-davinci-002",
        help="engine name of LLM",
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="whether to use retrieval-augmented KB-BINDER",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        metavar="N",
        default="item/GrailQA/grailqa_v1.0_train.json",
        help="training data path",
    )
    parser.add_argument(
        "--eva_data_path",
        type=str,
        metavar="N",
        default="data/GrailQA/grailqa_v1.0_dev.json",
        help="evaluation data path",
    )
    parser.add_argument(
        "--fb_roles_path",
        type=str,
        metavar="N",
        default="data/GrailQA/fb_roles",
        help="freebase roles file path",
    )
    parser.add_argument(
        "--surface_map_path",
        type=str,
        metavar="N",
        default="data/surface_map_file_freebase_complete_all_mention",
        help="surface map file path",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    nlp = spacy.load("en_core_web_sm")
    bm25_searcher = LuceneSearcher("contriever_fb_relation/index_relation_fb")
    query_encoder = AutoQueryEncoder(
        encoder_dir="PLMs/facebook/contriever", pooling="mean"
    )
    contriever_searcher = FaissSearcher(
        "contriever_fb_relation/freebase_contriever_index", query_encoder
    )
    hsearcher = HybridSearcher(contriever_searcher, bm25_searcher)

    # rela_corpus 和 bm25_searcher 一样 ？？？
    rela_corpus = LuceneSearcher("contriever_fb_relation/index_relation_fb")

    # process_file: 没做处理，直接返回了一个list
    dev_data = process_file(args.eva_data_path)
    train_data = process_file(args.train_data_path)
    que_to_s_dict_train = {
        data["question"]: data["s_expression"] for data in train_data
    }

    # debug
    # dev_data = dev_data[:]
    # train_data = train_data[:100]

    # process_file_node: 返回一个 q:mid list
    # e.g. 'who is playing bilbo baggins': {'m.0g6z1': 'bilbo baggins'}
    question_to_mid_dict = process_file_node(args.train_data_path)

    if not args.retrieval:
        (
            selected_quest_compose,
            selected_quest_compare,
            selected_quest,
        ) = select_shot_prompt_train(train_data, args.shot_num)
    else:
        selected_quest_compose = []
        selected_quest_compare = []
        selected_quest = []
    all_ques = selected_quest_compose + selected_quest_compare
    corpus = [data["question"] for data in train_data]
    tokenized_train_data = []
    for doc in corpus:
        nlp_doc = nlp(doc)
        tokenized_train_data.append([token.lemma_ for token in nlp_doc])
    bm25_train_full = BM25Okapi(tokenized_train_data)

    # prompt_type 举例
    """
    Question: what countries are the mediterranean
    Type of the question: Composition
    Question: what drugs does charlie sheen do
    Type of the question: Comparison
    Question: when did venus williams win wimbledon
    Type of the question: Comparison
    """
    if not args.retrieval:
        prompt_type = ""
        random.shuffle(all_ques)
        for que in all_ques:
            prompt_type = prompt_type + "Question: " + que + "\nType of the question: "
            if que in selected_quest_compose:
                prompt_type += "Composition\n"
            else:
                prompt_type += "Comparison\n"
    else:
        prompt_type = ""
    with open(args.fb_roles_path) as f:
        lines = f.readlines()
    relationships = []
    entities_set = []
    relationship_to_enti = {}
    for line in lines:
        info = line.split(" ")
        relationships.append(info[1])
        entities_set.append(info[0])
        entities_set.append(info[2])
        relationship_to_enti[info[1]] = [info[0], info[2]]

    name_to_id_dict = {}
    with open(args.surface_map_path) as f:
        # lines = f.readlines()
        for line in tqdm(
            f, desc="loading surface map", total=59956543, dynamic_ncols=True
        ):
            info = line.split("\t")
            name = info[0]
            score = float(info[1])
            mid = info[2].strip()
            if name in name_to_id_dict:
                name_to_id_dict[name][mid] = score
            else:
                name_to_id_dict[name] = {}
                name_to_id_dict[name][mid] = score

    all_fns = list(name_to_id_dict.keys())
    tokenized_all_fns = [fn.split() for fn in all_fns]
    bm25_all_fns = BM25Okapi(tokenized_all_fns)

    from multiprocessing.dummy import Pool
    import functools

    # all_combiner_evaluation(
    #     dev_data,
    #     selected_quest_compose,
    #     selected_quest_compare,
    #     selected_quest,
    #     prompt_type,
    #     hsearcher,
    #     rela_corpus,
    #     relationships,
    #     args.temperature,
    #     que_to_s_dict_train,
    #     question_to_mid_dict,
    #     args.    #     args.engine,
    #     name_to_id_dict,
    #     bm25_all_fns,
    #     all_fns,
    #     relationship_to_enti,
    #     retrieval=args.retrieval,
    #     corpus=corpus,
    #     nlp_model=nlp,
    #     bm25_train_full=bm25_train_full,
    #     retrieve_number=args.shot_num,
    # )

    mapper = functools.partial(
        all_combiner_evaluation,
        selected_quest_compose=selected_quest_compose,
        selected_quest_compare=selected_quest_compare,
        selected_quest=selected_quest,
        prompt_type=prompt_type,
        hsearcher=hsearcher,
        rela_corpus=rela_corpus,
        relationships=relationships,
        temp=args.temperature,
        que_to_s_dict_train=que_to_s_dict_train,
        question_to_mid_dict=question_to_mid_dict,
        LLM_engine=args.engine,
        name_to_id_dict=name_to_id_dict,
        bm25_all_fns=bm25_all_fns,
        all_fns=all_fns,
        relationship_to_enti=relationship_to_enti,
        retrieval=args.retrieval,
        corpus=corpus,
        nlp_model=nlp,
        bm25_train_full=bm25_train_full,
        retrieve_number=args.shot_num,
    )
    print("retrieval:",args.retrieval)
    pool = Pool(processes=100)
    pool.map(mapper, tqdm(dev_data))


if __name__ == "__main__":
    main()
