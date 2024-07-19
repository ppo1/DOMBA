import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, pipeline, set_seed
from datasets import load_dataset, Dataset
from peft import (
        get_peft_model,
        prepare_model_for_kbit_training,
        LoraConfig
    )
from trl import SFTTrainer
import os
import time
import math
from tqdm import trange, tqdm
import torch.nn.functional as F
from peft import PeftModel
from suffix_tree import Tree
import numpy as np

def get_at_inds(logits, inds):
    logits2d = logits.view(-1, logits.shape[-1])
    return logits2d[torch.arange(len(logits2d)), inds.view(-1)].view(inds.shape)

def log_exposure(logits1, logits2):
    rlp1 = logits2rel_log_prob(logits1)
    rlp2 = logits2rel_log_prob(logits2)
    return rlp1 - rlp2

def get_rank(logits, vals):
    vals = vals.view(*vals.shape, 1).expand(*logits.shape)
    return torch.sum(logits >= vals-0.001, dim=-1)

def logits2rel_log_prob(logits):
    m = torch.mean(logits, dim=-1, keepdim=True)
    return logits - m

def comb_logits(logits1, logits2, avg_func, relative=True):
    if relative:
        logits1 -= torch.mean(logits1, dim=-1, keepdim=True)
        logits2 -= torch.mean(logits2, dim=-1, keepdim=True)
    return avg_func(logits1, logits2)

def get_log_probs(logits, labels, ignore_index, reduction='none'):
    flat_labels = labels.view(-1)
    all_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), flat_labels, ignore_index=ignore_index, reduction='none')
    if reduction == 'none':
        return -all_loss.view(labels.shape)
    if reduction == 'mean':
        return torch.mean(-all_loss.view(labels.shape), dim=-1)

def print_eval(model_name, loss):
    print(f"\n{model_name} loss: {loss}. ppl: {math.exp(loss)}")

class ACModel(object):
    """A class for training and evaluating access controlled models"""
    def __init__(self, workdir, final_model_path, train_ds, test_ds, text_column, context_column, 
                mode='reg', r=64, context_format='{0}', base_model_path=None):

        self.workdir = workdir
        self.final_model_path = final_model_path
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.context_column = context_column
        self.text_column = text_column
        self.mode = mode
        self.r = r
        self.train_time = 0
        self.eval_tokens = None
        self.context_format = context_format
        self.base_model_path = base_model_path or "openai-community/openai-gpt"

    def run(self, num_train_epochs=1, eval_base=True, load_base=True):
        """Do all needed for training"""
        if load_base:
            self.load_base_model()
        if eval_base:
            base_loss,_ = self.eval_model('base')
            print_eval("base model", base_loss)
        if os.path.exists(self.final_model_path):
            print(f"peft model exists, loading from {self.final_model_path}")
            self.load_peft_model()
        else:
            self.init_peft_model()
            self.prepare_train_args(num_train_epochs)
            self.train_time = self.train()
            self.save_peft_model()
        peft_loss,_ = self.eval_model('peft')
        print_eval("peft model", peft_loss)
        return peft_loss

    def reg_formatting_func(self, data):
        output = []
        for text in data[self.text_column]:
            output.append(text)
        return output

    def load_peft_model(self):
        self.peft_model = PeftModel.from_pretrained(self.base_model, self.final_model_path, torch_dtype=torch.float16)

    def set_eval_tokens(self, sentences):
        """Set the tokens to evaluate (by default - all tokens)"""
        if sentences:
            self.eval_tokens = [list(self.tokenizer(' ' + s, return_tensors="pt").values())[0][0].to(self.base_model.device) for s in sentences]
            self.tokens_set = torch.concatenate(self.eval_tokens)
            self.tokens_tree = Tree({i: ['S'] + toks.tolist() + ['E'] for i,toks in enumerate(self.eval_tokens)})
        else:
            self.eval_tokens = None
            self.tokens_set = None
            self.tokens_tree = None

    def get_logits_and_labels(self, model_type='peft', dataset=None, i=0, only_eval_tokens=False):
        model = self.peft_model if model_type == 'peft' else self.base_model
        dataset = dataset or self.test_ds
        formatting_func = self.reg_formatting_func
        model.eval()
        torch.manual_seed(1234)
        toks = self.tokenizer(formatting_func(dataset[i:i+8]), return_tensors="pt", padding=True, truncation=True, max_length=512)
        toks = {k: v.to(model.device) for k, v in toks.items()}
        with torch.no_grad():
            output = model(**toks)
            logits = output[self.logits_key]
            labels = torch.roll(toks['input_ids'], shifts=-1, dims=1)
            if only_eval_tokens:
                labels_eval = torch.zeros(labels.shape)
                for i in range(len(labels)):
                    for j in range(len(labels[i])):
                        if labels[i][j] not in self.tokens_set:
                            continue
                        w = ['S', labels[i][j].item()]
                        length = 1
                        while self.tokens_tree.find(w):
                            ext_w = w + ['E']
                            if self.tokens_tree.find(ext_w):
                                labels_eval[i][j:j+length] = labels[i][j:j+length]
                            if j + length >= len(labels[i]):
                                break
                            w.append(labels[i][j+length].item())
                            length += 1
                return logits, labels, labels_eval
        return logits, labels

    def eval_model(self, model_type='peft', dataset=None, context=None):
        """Evaluates one sub model alone"""
        model = self.peft_model if model_type == 'peft' else self.base_model
        dataset = dataset or self.test_ds
        if context is not None:
            formatting_func = self.get_const_context_formatting_func(context)
        else:
            formatting_func = self.context_formatting_func if self.mode == 'context' else self.reg_formatting_func
        
        total_loss = 0
        count = 0
        if self.eval_tokens:
            loss_by_tok = [[0,0] for i in range(len(self.eval_tokens))]
        model.eval()
        torch.manual_seed(1234)
        for i in trange(0, len(dataset), 8):
            toks = self.tokenizer(formatting_func(dataset[i:i+8]), return_tensors="pt", padding=True, truncation=True, max_length=512)
            toks = {k: v.to(model.device) for k, v in toks.items()}
            with torch.no_grad():
                labels = torch.roll(toks['input_ids'], shifts=-1, dims=1)
                labels[:, -1] = len(self.tokenizer) - 1
                ignore_index=len(self.tokenizer) - 1
                if self.eval_tokens:
                    labels = labels * torch.isin(labels, self.tokens_set).to(model.device)
                    ignore_index = 0
                output = model(**toks)
                logits = output[self.logits_key]

                # Compute the loss
                flat_labels = labels.view(-1)
                
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), flat_labels, ignore_index=ignore_index)
                total_loss += loss.item()
                count += 1
            del toks, output, logits, labels, flat_labels

        if count == 0:
            return None, 0
        total_loss = total_loss / count
        return total_loss, count


    def eval_model_sup(self, sup_model, avg_funcs, model_type='peft', dataset=None, relative=True, base_model=None):
        """Evaluates the aggregation of 2 sub models"""
        model = self.peft_model if model_type == 'peft' else self.base_model
        dataset = dataset or self.test_ds
        formatting_func = self.reg_formatting_func
        
        total_loss = [0] * len(avg_funcs)
        count = [0] * len(avg_funcs)
        exposures = [[] for i in range(len(avg_funcs))]
        model.eval()
        sup_model.eval()
        if base_model:
            base_model.eval()
        torch.manual_seed(1234)
        res_log_ratio = [[] for i in range(len(avg_funcs))]
        for i in trange(0, len(dataset), 8):
            toks = self.tokenizer(formatting_func(dataset[i:i+8]), return_tensors="pt", padding=True, truncation=True, max_length=512)
            toks = {k: v.to(model.device) for k, v in toks.items()}
            with torch.no_grad():
                flat_toks = toks['input_ids'].view(-1)
                labels = torch.roll(toks['input_ids'], shifts=-1, dims=1)
                labels[:, -1] = len(self.tokenizer) - 1
                ignore_index = len(self.tokenizer) - 1
                if self.eval_tokens:
                    labels = labels * torch.isin(labels, self.tokens_set).to(model.device)
                    ignore_index = 0
                output = model(**toks)
                logits = output[self.logits_key]
                output_sup = sup_model(**toks)
                logits_sup = output_sup[self.logits_key]
                if base_model:
                    output_base = base_model(**toks)
                    logits_base = output_base[self.logits_key]
                flat_labels = labels.view(-1)

                for func_i, avg_func in enumerate(avg_funcs):
                    if "submix" in avg_func.__name__:
                        logits_avg = avg_func(logits, logits_sup, logits_base)
                    else:
                        logits_avg = comb_logits(logits, logits_sup, avg_func, relative)

                    log_probs_comb = get_log_probs(logits_avg, labels, ignore_index)
                    lexp_all = log_exposure(logits_avg, logits)
                    lexp_max = torch.max(lexp_all, dim=-1).values
                    lexp_true = get_at_inds(lexp_all, labels)

                    lexp_all_sup = log_exposure(logits_avg, logits_sup)
                    lexp_max_sup = torch.max(lexp_all_sup, dim=-1).values
                    lexp_true_sup = get_at_inds(lexp_all_sup, labels)

                    lexp_max_both = torch.maximum(lexp_max, lexp_max_sup)
                    lexp_true_both = torch.maximum(lexp_true, lexp_true_sup)
                    exposures[func_i].append((lexp_max_both, lexp_true_both, log_probs_comb))

                    # Compute the loss
                    if self.eval_tokens:
                        flat_logits_avg = logits_avg.view(-1, logits_avg.shape[-1])
                        flat_logits_base = logits_base.view(-1, logits_base.shape[-1])
                        all_loss = F.cross_entropy(flat_logits_avg, flat_labels, ignore_index=ignore_index, reduction='none')
                        inds = torch.nonzero(all_loss, as_tuple=True)[0].to('cpu')
                        if base_model:
                            all_loss_base = F.cross_entropy(flat_logits_base, flat_labels, ignore_index=ignore_index, reduction='none')
                        all_loss = all_loss.to('cpu')
                        flat_labels_cpu = flat_labels.to('cpu')
                        for j in inds:
                            if flat_labels_cpu[j] == ignore_index:
                                continue
                            w = ['S', flat_labels_cpu[j].item()]
                            length = 1
                            while self.tokens_tree.find(w):
                                ext_w = w + ['E']
                                if self.tokens_tree.find(ext_w):
                                    token_ind = self.tokens_tree.find_all(ext_w)[0][0]
                                    loss = sum(all_loss[j:j+length]) / length
                                    if base_model:
                                        loss_base = sum(all_loss_base[j:j+length]) / length
                                        rank = get_rank(flat_logits_avg[j], flat_logits_avg[j][flat_labels_cpu[j]])
                                        rank_base = get_rank(flat_logits_base[j], flat_logits_base[j][flat_labels_cpu[j]])
                                        best = torch.argmax(flat_logits_avg[j])
                                        rank_base_best = get_rank(flat_logits_base[j], flat_logits_base[j][best])
                                        res_log_ratio[func_i].append((w[1:],((loss_base - loss).item(), rank.item(), rank_base.item(), rank_base_best.item()), i, flat_toks[j-3:j].cpu()))
                                    total_loss[func_i] += loss
                                    count[func_i] += 1
                                if j + length >= len(flat_labels_cpu):
                                    break
                                w.append(flat_labels_cpu[j+length].item())
                                length += 1
                        del all_loss
                    else:
                        loss = F.cross_entropy(logits_avg.view(-1, logits_avg.shape[-1]), flat_labels, ignore_index=ignore_index)
                        total_loss[func_i] += loss
                        count[func_i] += 1
            # del toks, output, logits_avg, labels, flat_labels

        exp_res = []
        for exps in exposures:
            lp = torch.concatenate([r[2].view(-1) for r in exps])
            exp_max = torch.concatenate([r[0].view(-1) for r in exps])[lp < 0]
            exp_true = torch.concatenate([r[1].view(-1) for r in exps])[lp < 0]
            if len(exp_max) == 0:
                print("no exposures. len orig:", len(lp))
                exp_res.append((0,0,0,0,0,0,0))
                continue
            k_50 = max(1,int(len(exp_max)*0.5))
            k_99 = max(1,int(len(exp_max)*0.99))
            exp_50_max = torch.kthvalue(exp_max, k_50).values.item()
            exp_50_true = torch.kthvalue(exp_true, k_50).values.item()
            exp_99_max = torch.kthvalue(exp_max, k_99).values.item()
            exp_99_true = torch.kthvalue(exp_true, k_99).values.item()
            exp_max_max = torch.max(exp_max).item()
            exp_max_true = torch.max(exp_true).item()
            exp_res.append((exp_50_max, exp_99_max, exp_max_max, exp_50_true, exp_99_true, exp_max_true, len(exp_max)))

        if count[0] == 0:
            return None, 0
        res = [[(l/c).item(),c,er] for l,c,er in zip(total_loss, count, exp_res)]
        if self.eval_tokens and base_model:
            res = [r + [lr] for r,lr in zip(res, res_log_ratio)]
        return res

    def save_peft_model(self):
        print(f"saving peft model to {self.final_model_path}")
        self.peft_model.save_pretrained(self.final_model_path, save_embedding_layers=False)

    def merge_and_save_model(self):
        merged_model = self.peft_model.merge_and_unload()
        path = self.final_model_path + "_full"
        print(f"saving merged model to {path}")
        merged_model.save_pretrained(path)

    def load_base_model(self, add_pad=True):
        self.base_model = OpenAIGPTLMHeadModel.from_pretrained(self.base_model_path).to("cuda")
        self.tokenizer  = OpenAIGPTTokenizer.from_pretrained("openai-community/openai-gpt")
        self.logits_key = 'logits'
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.base_model.resize_token_embeddings(len(self.tokenizer))

    def init_peft_model(self):
        self.lora_config = LoraConfig(
            r=self.r,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['lm_head', 'c_attn', 'c_proj', 'c_fc']
        )
        self.peft_model = prepare_model_for_kbit_training(self.base_model)
        self.peft_model = get_peft_model(self.peft_model, self.lora_config)

    def prepare_train_args(self, num_train_epochs=1):
        per_device_train_batch_size = 16
        gradient_accumulation_steps = 4
        per_device_eval_batch_size = 16
        eval_accumulation_steps = 4
        optim = "paged_adamw_32bit"
        save_steps = 50
        logging_steps = 50
        learning_rate = 5e-4
        max_grad_norm = 0.3
        warmup_ratio = 0.03
        eval_strategy="steps"
        lr_scheduler_type = "constant"

        self.training_args = transformers.TrainingArguments(
                    output_dir=self.workdir,
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    optim=optim,
                    eval_strategy=eval_strategy,
                    save_steps=save_steps,
                    learning_rate=learning_rate,
                    logging_steps=logging_steps,
                    max_grad_norm=max_grad_norm,
                    num_train_epochs=num_train_epochs,
                    warmup_ratio=warmup_ratio,
                    group_by_length=True,
                    lr_scheduler_type=lr_scheduler_type,
                    ddp_find_unused_parameters=False,
                    eval_accumulation_steps=eval_accumulation_steps,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                )

    def train(self):
        formatting_func = self.context_formatting_func if self.mode == 'context' else self.reg_formatting_func
        print(f"using {formatting_func.__name__} for formatting")
        self.trainer = SFTTrainer(
            model=self.peft_model,
            train_dataset=self.train_ds,
            eval_dataset=self.test_ds,
            peft_config=self.lora_config,
            formatting_func=formatting_func,
            max_seq_length=512,
            tokenizer=self.tokenizer,
            args=self.training_args,
        )

        # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
        for name, module in self.trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

        t0 = time.time()
        self.trainer.train()
        self.trainer.save_model(f"{self.workdir}/final")
        train_time = (time.time() - t0) / 60
        print(f"train time: {train_time} minutes")
        return train_time

