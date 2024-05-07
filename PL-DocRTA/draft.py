import json
from prompt_generation import prepro
import random

random_number = random.uniform(40, 50)
print(random_number)
test_data = []
train_data = []
# prompt_model.load_state_dict(torch.load(f"{args.project_root}/ckpts/{this_run_unicode}.ckpt"))
with open("./datasets/wiki_time/train.json", "r") as file:
    json_data = file.read()
    train_data = json.loads(json_data)

with open("./datasets/wiki_time/test.json", "r") as file:
    json_data = file.read()
    test_data = json.loads(json_data)


# data = train_data + test_data
# print("文档数目", len(data))
# print("训练集文档数目", len(train_data))
# print("测试集文档数目", len(test_data))

# t_number = 0

# for doc in test_data:
#     t_number += len(doc['labels'])
# print("测试集实体对数目", t_number)

t_number = 0
time_begin = []
time_end = []
for doc in test_data:
    for label in doc["labels"]:
        time_begin.append(label['begin_time'])
        time_end.append(label['end_time'])

begin_cor = 0
end_cor = 0
all_cor = 0
i = 0
with open("./datasets/wiki_time/generate_False.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        strs = line.split(' ')
        # print(strs)
        b_flag = False
        e_flag = False
        if strs[1] == time_begin[i]:
            begin_cor += 1
            b_flag = True
        if strs[3] == time_end[i] or (strs[3] == strs[1] and time_end[i] == "NA"):
            end_cor += 1
            e_flag = True
        i += 1
        if b_flag and e_flag:
            all_cor += 1
print("开始时间正确{}，结束时间正确{}，全都正确{}".format(begin_cor/i, end_cor/i, all_cor/i))

# rel_dic = []
# for doc in data:
#     for label in doc['labels']:
#         if label['r'] not in rel_dic:
#             rel_dic.append(label['r'])



