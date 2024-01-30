import json
from time import sleep

import os
from tqdm import tqdm


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


from metaqa_1hop import client, LLM_engine


def type_process(file_name):
    test_type_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            type = line.strip()
            test_type_list.append(type)
    return test_type_list


# test_type_list_1hop = type_process('data/metaQA/qa_test_qtype_1hop.txt')
# train_type_list_1hop = type_process('data/metaQA/qa_train_qtype_1hop.txt')


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
    prompt = "Given the following operations: actor_to_movie, movie_to_writer, tag_to_movie, writer_to_movie, movie_to_year, director_to_movie, movie_to_language, movie_to_genre, movie_to_director, movie_to_actor, movie_to_tags\nQuestion: the films that share actors with the film [Dil Chahta Hai] were released in which years\nLogical Form: movie_to_year(actor_to_movie(movie_to_actor([Dil Chahta Hai])))\nThree operations: movie_to_year, actor_to_movie, movie_to_actor\nQuestion: who are the directors of the movies written by the writer of [The Green Mile]\nLogical Form: movie_to_director(writer_to_movie(movie_to_writer([The Green Mile])))\nThree operations: movie_to_director, writer_to_movie, movie_to_writer\nQuestion: what types are the films directed by the director of [For Love or Money]\nLogical Form: movie_to_genre(director_to_movie(movie_to_director([For Love or Money])))\nThree operations: movie_to_genre, director_to_movie, movie_to_director\nQuestion: when did the movies release whose actors also appear in the movie [Cast a Deadly Spell]\nLogical Form: movie_to_year(actor_to_movie(movie_to_actor([Cast a Deadly Spell])))\nThree operations: movie_to_year, actor_to_movie, movie_to_actor\n".replace("[","").replace("]","")
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

    test_question_3hop = ques_ans_process("data/metaQA/qa_test_3hop.txt")

    # small data
    test_data = json.load(open("../LLM_KGQA/data/metaqa/test/3-hop.json"))
    valid_questions_map = {d["question"]:d["id"] for d in test_data}
    for d in test_question_3hop:
        d["question"] = d["question"].replace("[", "").replace("]", "")
    new_test_question = []
    for d in test_question_3hop:
        if d["question"] in valid_questions_map:
            d["id"] = valid_questions_map[d["question"]]
            new_test_question.append(d)
    assert len(new_test_question) == 300
    test_question_3hop = new_test_question

    type_to_ques_dict = {}

    total = 0
    correct = 0

    out = f"save/metaqa/3-hop/KB-BINDER-{LLM_engine}"
    os.makedirs(out, exist_ok=True)

    def _func(item):
        if os.path.exists(f"{out}/{item['id']}.json"):
            return  
        question = item["question"]
        print("question: ", question, flush=True)
        got_result = False
        while got_result is not True:
            try:
                question_type,usage = two_hop_type_generator(question)
                # print("question_type: ", question_type)
                question_type = question_type.split("operations: ")[1]
                question_type = question_type.split(", ")
                print("question_type: ", question_type)
                if len(question_type) < 3:
                    break
                rela_0 = question_type[0]
                rela_1 = question_type[1]
                rela_2 = question_type[2]
                got_result = True
            except Exception as e:
                print("error in type generation", e)
                sleep(3)
        question_type.reverse()
        print("question_type: ", question_type, flush=True)
        
        try:
            if len(question_type) < 3:
                set_pred = set()
            else:
                ent = item["retrieved_ent"]
                first_step_ans = retrieve_answer(question_type[0], ent)
                first_step_ans = list(set(first_step_ans))
                if ent in first_step_ans:
                    first_step_ans.remove(ent)
                print("first_step_ans: ", first_step_ans, flush=True)
                second_step_ans = []
                for ent_mid in first_step_ans:
                    second_step_ans = second_step_ans + retrieve_answer(
                        question_type[1], ent_mid
                    )
                second_step_ans = list(set(second_step_ans))
                if ent in second_step_ans:
                    second_step_ans.remove(ent)
                # print("second_step_ans: ", second_step_ans, flush=True)
                pred = []
                for ent_mid in second_step_ans:
                    pred = pred + retrieve_answer(question_type[2], ent_mid)
                # print("answer: ", item["answer"], flush=True)
                # print("pred: ", list(set(pred)))
                set_pred = set(pred)
            if ent in set_pred:
                set_pred.remove(ent)
            # if set_pred == set(item["answer"]):
            #     correct += 1
            # total += 1
            # print("total: ", total, flush=True)
            # print("correct: ", correct, flush=True)
            # print("accuracy: ", correct / total, flush=True)
        except Exception as e:
            print("error in prediction", e)
            pred = None
        # save
        item["prediction"] = pred
        json.dump(item, open(f"{out}/{item['id']}.json", "w"), indent=4, ensure_ascii=False)

    data =  tqdm(test_question_3hop)
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(1)
    pool.map(_func, data)
