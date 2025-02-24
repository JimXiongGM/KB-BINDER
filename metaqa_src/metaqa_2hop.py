import json
from time import sleep

import os
from tqdm import tqdm

import spacy

nlp = spacy.load("en_core_web_sm")
import re

import numpy as np
from bm25_trial import BM25_self


from metaqa_1hop import client, LLM_engine

def type_process(file_name):
    test_type_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            type = line.strip()
            test_type_list.append(type)
    return test_type_list



def ques_ans_process(file_name):
    return_dict_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            return_dict = {}
            question, answers = line.split("\t")
            answer_list = answers.split("|")
            answer_list[-1] = answer_list[-1].strip()
            ent_s_idx = question.index("[")
            ent_e_idx = question.index("]")
            retrieved_ent = question[ent_s_idx + 1 : ent_e_idx]
            return_dict["question"] = question
            return_dict["retrieved_ent"] = retrieved_ent
            return_dict["answer"] = answer_list
            return_dict_list.append(return_dict)
    return return_dict_list


def two_hop_type_generator(question):
    prompt = "Given the following operations: actor_to_movie, movie_to_writer, tag_to_movie, writer_to_movie, movie_to_year, director_to_movie, movie_to_language, movie_to_genre, movie_to_director, movie_to_actor, movie_to_tags\nQuestion: which person wrote the films directed by [Yuriy Norshteyn]\nLogical Form: movie_to_writer(director_to_movie([Yuriy Norshteyn]))\nTwo operations: movie_to_writer, director_to_movie\nQuestion: which movies have the same director of [Just Cause]\nLogical Form: director_to_movie(movie_to_director([Yuriy Norshteyn]))\nTwo operations: director_to_movie, movie_to_director\nQuestion: what genres do the movies written by [Maureen Medved]\nLogical Form: movie_to_genre(writer_to_movie([Maureen Medved]))\nTwo operations: movie_to_genre, writer_to_movie\nQuestion: what were the release years of the movies acted by [Todd Field]\nLogical Form: movie_to_year(actor_to_movie([Todd Field]))\nTwo operations: movie_to_year, actor_to_movie\nQuestion: the films written by [Babaloo Mandel] starred which actors\nLogical Form: movie_to_actor(writer_to_movie([Babaloo Mandel]))\nTwo operations: movie_to_actor, writer_to_movie\n".replace("[","").replace("]","")
    prompt = prompt + "Question: " + question + "\nLogical Form: "

    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": prompt},
    ]
    while 1:
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



def retrieve_answer(found_type, found_ent):
    found_relas = enti_to_fact_dict[found_ent]
    # print("found_ent: ", found_ent)
    # print("found_relas: ", found_relas)
    rela_to_ans_dict = {}
    for fact in found_relas:
        s, r, o = fact.split("|")
        if s == found_ent:
            if r in rela_to_ans_dict:
                rela_to_ans_dict[r].append(o)
            else:
                rela_to_ans_dict[r] = [o]
        if o == found_ent:
            if r in rela_to_ans_dict:
                rela_to_ans_dict[r].append(s)
            else:
                rela_to_ans_dict[r] = [s]
    rela = type_to_rela_dict[found_type]
    # print("rela_to_ans_dict: ", rela_to_ans_dict)
    if rela in rela_to_ans_dict:
        return rela_to_ans_dict[rela]
    else:
        return []


if __name__ == "__main__":
    enti_to_fact_dict = {}
    with open("data/metaQA/kb.txt") as f:
        lines = f.readlines()
        for line in lines:
            s, r, o = line.split("|")
            if s.strip() not in enti_to_fact_dict:
                enti_to_fact_dict[s.strip()] = [line.strip()]
            else:
                enti_to_fact_dict[s.strip()].append(line.strip())
            if o.strip() not in enti_to_fact_dict:
                enti_to_fact_dict[o.strip()] = [line.strip()]
            else:
                enti_to_fact_dict[o.strip()].append(line.strip())

    test_question_2hop = ques_ans_process("data/metaQA/qa_test_2hop.txt")

    # small data
    test_data = json.load(open("../LLM_KGQA/data/metaqa/test/2-hop.json"))
    valid_questions_map = {d["question"]:d["id"] for d in test_data}
    for d in test_question_2hop:
        d["question"] = d["question"].replace("[", "").replace("]", "")
    new_test_question = []
    for d in test_question_2hop:
        if d["question"] in valid_questions_map:
            d["id"] = valid_questions_map[d["question"]]
            new_test_question.append(d)
    assert len(new_test_question) == 300
    test_question_2hop = new_test_question

    type_to_rela_dict = {
        "tag_to_movie": "has_tags",
        "writer_to_movie": "written_by",
        "movie_to_tags": "has_tags",
        "movie_to_year": "release_year",
        "movie_to_writer": "written_by",
        "movie_to_language": "in_language",
        "movie_to_genre": "has_genre",
        "director_to_movie": "directed_by",
        "movie_to_actor": "starred_actors",
        "movie_to_director": "directed_by",
        "actor_to_movie": "starred_actors",
    }
    types_all = list(type_to_rela_dict.keys())
    types_all_spl = [type_.split("_") for type_ in types_all]
    type_drops_all = []
    for i, rela in enumerate(types_all_spl):
        drops_types_all = []
        for word in rela:
            doc = nlp(word)
            if len(doc) > 0:
                drops_types_all.append(doc[0].lemma_)
        type_drops_all.append(" ".join(drops_types_all))
    print("type_drops_all: ", type_drops_all)
    bm25_all_relas = BM25_self()
    bm25_all_relas.fit(type_drops_all)

    total = 0
    correct = 0

    out = f"save/metaqa/2-hop/KB-BINDER-{LLM_engine}"
    os.makedirs(out, exist_ok=True)

    for item in tqdm(test_question_2hop):
        if os.path.exists(f"{out}/{item['id']}.json"):
            continue    
        question = item["question"]
        print("question: ", question)
        got_result = False
        while got_result is not True:
            try:
                question_type,usage = two_hop_type_generator(question)
                # print("question_type: ", question_type)
                question_type = question_type.split("operations: ")[1]
                question_type = question_type.split(", ")
                for idx, type_ in enumerate(question_type):
                    if type_ not in type_to_rela_dict:
                        tokenized_query = re.split("_", type_)
                        tokenized_ques = question.split()
                        tokenized_query = tokenized_query + tokenized_ques
                        drops_query = []
                        for word in tokenized_query:
                            doc = nlp(word)
                            if len(doc) > 0:
                                drops_query.append(doc[0].lemma_)
                        drops_query = " ".join(drops_query)
                        scores = list(
                            bm25_all_relas.transform(
                                drops_query, [i for i in range(11)]
                            )
                        )
                        sorted_score_index = list(np.argsort(scores))
                        sorted_score_index.reverse()
                        bound_type = sorted_score_index[0]
                        question_type[idx] = types_all[bound_type]
                got_result = True
            except:
                sleep(3)
        question_type.reverse()
        print("question_type: ", question_type)
        # relas = type_to_rela_dict[question_type]
        # print("relas: ", relas)
        ent = item["retrieved_ent"]
        first_step_ans = retrieve_answer(question_type[0], ent)
        print("first_step_ans: ", first_step_ans)
        pred = []
        for ent_mid in first_step_ans:
            pred = pred + retrieve_answer(question_type[1], ent_mid)
        print("answer: ", item["answer"])
        print("pred: ", list(set(pred)))
        set_pred = set(pred)
        if ent in set_pred:
            set_pred.remove(ent)
        if set_pred == set(item["answer"]):
            correct += 1
        total += 1
        print("total: ", total)
        print("correct: ", correct)
        print("accuracy: ", correct / total)
    
        # save
        item["prediction"] = pred
        json.dump(item, open(f"{out}/{item['id']}.json", "w"), indent=4, ensure_ascii=False)
