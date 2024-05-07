from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
import os
import sys
import ast
import http.client
import json

os.environ["http_proxy"] = "http://10.201.201.181:7890"
os.environ["https_proxy"] = "http://10.201.201.181:7890"

def century_prompt(dataset):
    classes = [
        "18",
        "19",
        "20"
    ]
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "18": ["18th century", "1800s"],
            "19": ["19th century", "1900s"],
            "20": ["20th century", "2000s"]
        },
        tokenizer = tokenizer,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} , As we can see from the above , {"placeholder":"text_b"} in the {"mask"}',
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    century_answer = []
    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            century_answer.append(str(classes[preds]))
            # print("century" + str(classes[preds]))
    return century_answer

def age_prompt(dataset):
    classes = [ 
        "0","1","2","3","4","5","6","7","8","9"
    ]
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "0": ["00s", "aughts"],
            "1": ["10s", "tens"],
            "2": ["20s", "twenties"],
            "3": ["30s", "thirties"],
            "4": ["40s", "forties"],
            "5": ["50s", "fifties"],
            "6": ["60s", "sixties"],
            "7": ["70s", "seventies"],
            "8": ["80s", "eighties"],
            "9": ["90s", "nineties"],
        },
        tokenizer = tokenizer,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} , As we can see from the above , {"placeholder":"text_b"} in the {"mask"}',
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    age_answer = []

    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            age_answer.append(str(classes[preds]))
            # print("age" + str(classes[preds]))
    # predictions would be 1, 0 for classes 'positive', 'negative'
    return age_answer

def year_prompt(dataset):
    classes = []
    for i in range(1950,2000,1):
        classes.append(str(i))
    label_words = {item : [item] for item in classes}
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = label_words,
        tokenizer = tokenizer,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text = 'Obama assumed the presidency of the United States in 2008, digit 2008 . {"placeholder":"text_a"} , digit {"mask"}',
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    year_answer = []

    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            year_answer.append(str(classes[preds]))
            # print("age" + str(classes[preds]))
    # predictions would be 1, 0 for classes 'positive', 'negative'
    return year_answer

def month_prompt(dataset):
    classes = [ 
            "1","2","3","4","5","6","7","8","9","10","11","12"
    ]
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "1": ["January", "tens"],
            "2": ["February", "twenties"],
            "3": ["March", "thirties"],
            "4": ["April", "forties"],
            "5": ["May", "fifties"],
            "6": ["June", "sixties"],
            "7": ["July", "seventies"],
            "8": ["August", "eighties"],
            "9": ["September", "nineties"],
            "10":["October"],
            "11":["November"],
            "12":["December"]
        },
        tokenizer = tokenizer,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} , As we can see from the above , {"placeholder":"text_b"} in month {"mask"}',
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )

    age_answer = []

    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim = -1)
            age_answer.append(str(classes[preds]))
            # print("age" + str(classes[preds]))
    # predictions would be 1, 0 for classes 'positive', 'negative'
    return age_answer

def gpt_request(prompt):
    token = "sk-5BulJkagVsEL5HlN1cC8D3A97fFc400782Bf4438Ba3005A0"
    conn = http.client.HTTPSConnection("api.aigcbest.top")
    payload = json.dumps({
        "model": "gpt-3.5-turbo-instruct",
        "prompt": prompt,
        "max_tokens": 20,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "logprobs": None,
        "stop": "."
    })
    headers = {
    'Authorization': 'Bearer ' + token,
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    data = json.loads(data.decode("utf-8"))
    return data["choices"][0]["text"]


def main():
    dataset = []
    filename = "wiki_time/train.txt"
    answer = []
    train_number = 10
    with open(os.path.join(sys.path[0], filename), 'r', encoding='utf-8') as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            year_time_int = int(data["time"][0:4])

            if year_time_int > 2000 or year_time_int < 1950:
                continue

            train_number -= 1
            if train_number == 0:
                break
            text_b = data["en1"].replace('_',' ') + " " +  data["relation"].replace('_',' ') + " " + data["en2"].replace("_", " ")
            text_a = data["sent"]
            input_text = InputExample(guid=data["id"], text_a = text_a, text_b = text_b)
            answer.append(data["time"])
            dataset.append(input_text)

    # print(dataset)
    # age_answer = age_prompt(dataset)
    # century_answer = century_prompt(dataset)
    year_answer = year_prompt(dataset)
    predicate_answer = year_answer
    true_number = 0
    for i in range(len(answer)):
        if answer[i][0:4] == predicate_answer[i]:
            true_number += 1
        else:
            print("%s,%s" %(predicate_answer[i], dataset[i]))
    # true_number = [ a[0:4] == b for a,b in zip(answer, predicate_answer)]
    print("总数：%d，正确：%d，百分比：%f" %(len(dataset), true_number, true_number/len(dataset)))

def prepro(file_name):
    data_json = []
    dataset = []
    with open("./datasets/wiki_time/" + file_name, "r") as file:
        json_data = file.read()
        data_json = json.loads(json_data)
        guid = 0
        for doc in data_json:
            sentence = ""
            for sent in doc["sents"]:
                sentence += ' '.join(sent)
            for label in doc["labels"]:
                h = label["h"].replace("_", " ")
                r = label["r"].replace("_", " ")
                t = label["t"].replace("_", " ")
                text_b = h + " " + r + " " + t
                tgt_text = " from " + label["begin_time"] + " to " + label["end_time"]
                data_case = {"text_a":sentence, "text_b":text_b, "tgt_text":tgt_text, "guid":guid}
                guid += 1
                dataset.append(data_case)
    
    return dataset

def template(text_a, text_b, mode):
    fill = "Example:Fill in the [MASK] position. From document In 1947-1-1 Eugenia Livanos married Stavros Niarchos, and they divorced in 1990-1-1, we know that Eugenia Livanos spouse Stavros Niarchos exists at the beginning and end times of [MASK]. Answer:From 1947-1-1 to 1990-1-1. Question:Fill in the [MASK] position. "
    prompt_template = {
        "a":"From document [d], we know that [t] exists at the beginning and end times of [MASK]",
        "b":"According to the description in the document [d], we can determine that the entity relationship triplet [t] exists for [MASK]",
        "c":"A relational extraction task extracts the triplet [t] and its time range [MASK] from the document [d]"
    }

    return fill + prompt_template[mode].replace("[d]", text_a).replace("[t]", text_b) + ". Answer: "


def main_gpt():
    dataset = prepro("test.json")

    answer = []
    for data in dataset[:100]:
        text = template(data["text_a"], data["text_b"], "a")
        answer.append(gpt_request(text))
    
    with open("./datasets/wiki_time/gpt3.5_100.txt",'w') as f:
        for i in answer:
            f.write(i+"\n")
    # predicate_answer = year_answer
    # true_number = 0
    # for i in range(len(answer)):
    #     if answer[i][0:4] == predicate_answer[i]:
    #         true_number += 1
    #     else:
    #         print("%s,%s" %(predicate_answer[i], dataset[i]))
    # # true_number = [ a[0:4] == b for a,b in zip(answer, predicate_answer)]
    # print("总数：%d，正确：%d，百分比：%f" %(len(dataset), true_number, true_number/len(dataset)))

if __name__ == "__main__":
    # main()
    # gpt_request("In January 01, 1988, Mike Patton became the lead vocalist for San Francisco's Faith.Mike Patton became the lead vocalist for San Francisco's Faith in ")
    main_gpt()


