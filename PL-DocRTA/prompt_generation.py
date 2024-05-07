import torch
import json
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.utils.metrics import generation_metric
from openprompt import PromptForGeneration
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader
from transformers import AdamW
import argparse
from transformers.optimization import get_linear_schedule_with_warmup



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
                text_b = label["h"] + " " + label["r"] + " " + label["t"]
                tgt_text = " from " + label["begin_time"] + " to " + label["end_time"]
                data_case = InputExample(guid=guid, text_a=sentence, text_b=text_b, tgt_text=tgt_text)
                guid += 1
                dataset.append(data_case)
    
    return dataset

def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    begin_cor = 0
    end_cor = 0
    all_cor = 0
    tol = 0
    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)

        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        
        for i in range(len(output_sentence)):
            labels = inputs['tgt_text'][i].strip().split(' ')
            predicts = output_sentence[i].strip().split(' ')

            b_flag = False
            e_flag = False
            if predicts[1] == labels[1]:
                begin_cor += 1
                b_flag = True
            if predicts[3] == labels[3] or (predicts[3] == predicts[1] and labels[3] == "NA"):
                end_cor += 1
                e_flag = True
            tol += 1
            if b_flag and e_flag:
                all_cor += 1
    print("开始时间正确{}，结束时间正确{}，全都正确{}".format(begin_cor/tol, end_cor/tol, all_cor/tol))
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--plm_eval_mode", action="store_true")
    parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
    parser.add_argument("--model_name_or_path", default='t5-base')
    parser.add_argument("--device", default='0')
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--decoder_max_length", type=int, default=256)
    parser.add_argument("--save_path", default="./checkpoints/t5-base.ckpt")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--using_decoder_past_key_values", default="true")
    args = parser.parse_args()
    if args.using_decoder_past_key_values == "true":
        args.using_decoder_past_key_values = True
    else:
        args.using_decoder_past_key_values = False
    dataset = {}
    dataset["train"] = prepro("train.json")
    dataset["test"] = prepro("test.json")
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    generation_arguments = {
        "max_length": 512,
        "max_new_tokens": None,
        "min_length": 5,
        "temperature": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "num_beams": 5,
        "bad_words_ids": [[628], [198]]
    }
    
    mytemplate = PrefixTuningTemplate(model=plm, mid_dim = 512, tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"placeholder":"text_b"} {"mask"} ', using_decoder_past_key_values=args.using_decoder_past_key_values)


    
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length,
        batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True, 
        truncate_method="head")

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length,
        batch_size=args.batch_size,shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")


    prompt_model = PromptForGeneration(plm=plm, template=mytemplate, freeze_plm=False, tokenizer=tokenizer, plm_eval_mode=False)
    if args.mode == "eval":
        prompt_model.load_state_dict(torch.load(args.save_path))
    prompt_model = prompt_model.cuda()

    use_cuda = True

    prompt_model.to(int(args.device))
    if args.mode == "train":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
        ]


        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

        tot_step  = len(train_dataloader)*5
        global_step = 0
        tot_loss = 0
        log_loss = 0
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

        for epoch in range(args.epoch):
            prompt_model.train()
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                global_step +=1
                loss = prompt_model(inputs)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if global_step %50 ==0:
                    print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/50, scheduler.get_last_lr()[0]), flush=True)
                    log_loss = tot_loss
            torch.save(prompt_model.state_dict(),args.save_path)
    
    if args.mode == "eval":
        generated_sentence = evaluate(prompt_model, test_dataloader)
        with open(f"./datasets/wiki_time/generate_{args.epoch}_{args.model}.txt",'w') as f:
            for i in generated_sentence:
                f.write(i+"\n")


