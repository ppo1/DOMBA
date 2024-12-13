import pickle
import os
from datasets import load_dataset, Dataset
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import numpy as np
import random
import acmodel
from tqdm import trange, tqdm
from suffix_tree import Tree
import torch
import scipy.stats
from pandas import DataFrame
import pandas as pd
from datetime import datetime

def add2csv(data, path):
    """
    Add rows to csv to aggregate results.
    """
    df = DataFrame.from_records(data)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df])
    df.to_csv(path, index=False)
    return df

class Evaluator(object):
    """
    A class used to evaluate DOMBA
    Metrics:
    - perplexity 
    - exposure
    - secret inference attack
    - canary attack
    """
    def __init__(self, access_levels, ac_col, al2secrets, secret2rep, text_column, text_fil_column, ds_keys, workdir):
        self.access_levels = access_levels
        self.ac_col = ac_col
        self.al2secrets = al2secrets
        self.secret2rep = secret2rep
        self.text_column = text_column
        self.text_fil_column = text_fil_column
        self.ds_keys = ds_keys
        self.workdir = workdir
        if not os.path.exists(workdir):
            os.mkdir(workdir)

    def delete(self, exp_name):
        res_path = os.path.join(self.workdir, exp_name + ".pkl")
        if os.path.exists(res_path):
            os.remove(res_path)
        else:
            print("path doesn't exist")

    def get_path(self, exp_name, ext):
        return os.path.join(self.workdir, exp_name + "_" + ext + ".pkl")

    def get_eval_res(self, exp_name):
        res_path = self.get_path(exp_name, "eval")
        if os.path.exists(res_path):
            return pickle.load(open(res_path,"rb"))
        else:
            print(f"path doesn't exist: {res_path}")
            return None

    def get_eval_file_path(self, exp_name, func_name, num_funcs, fil):
        ext = ""
        if num_funcs > 1:
            ext += f"_{func_name}"
        if fil:
            ext += "_fil"
        ext += "_eval"
        ext = ext[1:]
        return self.get_path(exp_name, ext)

    def eval(self, acms1, acms2, avg_funcs, ds, exp_name, ref_model=None, fil=False, 
                overide=False, only_secrets=True, load_models=True, next_al=False):
        """
        Runs evaluation and saves results.
        To be later parsed by the different metrics calculation methods.
        """
        num_funcs = len(avg_funcs)
        if not overide:
            avg_funcs_to_update = []
            for f in avg_funcs:
                path = self.get_eval_file_path(exp_name, f.__name__, num_funcs, fil)
                if not os.path.exists(path):
                    avg_funcs_to_update.append(f)
            avg_funcs = avg_funcs_to_update
        if not avg_funcs:
            return

        if load_models:
            if type(acms1) == dict:
                for acm1 in acms1.values():
                    acm1.load_base_model()
                    acm1.load_peft_model()
                    acm1.load_base_model()
                for acm2 in acms2.values():
                    acm2.load_base_model()
                    acm2.load_peft_model()
                    acm2.load_base_model()
            else:  
                acm1 = acms1
                acm2 = acms2
                acm1.load_base_model()
                acm1.load_peft_model()
                acm1.load_base_model()
                acm2.load_base_model()
                acm2.load_peft_model()
                acm2.load_base_model()

        
        eval_res = {f.__name__:{} for f in avg_funcs}
        for i, al in enumerate(self.access_levels):
            print(f"{i}/{len(self.access_levels)}")
            nal = self.access_levels[(i+1)%len(self.access_levels)]
            al_ds = Dataset.from_list([x for x in ds if al == x[self.ac_col]])
            if len(al_ds) == 0:
                print(f"empty access level: {al}")
                continue

            if type(acms1) == dict:
                if next_al:
                    acm1 = acms1[nal]
                    acm2 = acms2[nal]
                else:
                    acm1 = acms1[al]
                    acm2 = acms2[al]
            else:
                acm1 = acms1
                acm2 = acms2

            curr_ref_model = ref_model or acm1.base_model

            if fil:
                acm1.text_column = self.text_fil_column
                secrets = [self.secret2rep[s] for s in self.al2secrets[al] if s in self.secret2rep]
            else:
                acm1.text_column = self.text_column
                secrets = self.al2secrets[al] 
            if only_secrets:
                acm1.set_eval_tokens(secrets)
            else:
                acm1.set_eval_tokens([])
            al_res = acm1.eval_model_sup(acm2.peft_model, avg_funcs, dataset=al_ds, base_model=curr_ref_model)
            for func, f_al_res in zip(avg_funcs, al_res):
                eval_res[func.__name__][al] = f_al_res
        
        if not any(eval_res.values()):
            print("Warning: No results")
            return
        for f_name, eres in eval_res.items():
            path = self.get_eval_file_path(exp_name, f_name, num_funcs, fil)
            pickle.dump(eres, open(path,"wb"))

    def ppl(self, exp_name, ep_mod, csv_name, field_pref=""):
        all_eval_res = self.get_eval_res(exp_name)
        total_loss = np.mean([x[0] for x in all_eval_res.values() if x])
        total_count = sum([x[1] for x in all_eval_res.values() if x])
        return add2csv([{"experiment":ep_mod, field_pref+"loss": total_loss, field_pref+"PPL": np.exp(total_loss), field_pref+"count": total_count}], 
                os.path.join(self.workdir,csv_name + ".csv"))

    def exposure_eval(self, exp_name, ep_mod, csv_name):
        all_eval_res = self.get_eval_res(exp_name)
        total_exp = [np.mean([np.exp(x[2][i]) for x in all_eval_res.values() if x]) for i in range(6)]
        total_len = sum([x[2][-1] for x in all_eval_res.values() if x])

        return add2csv([{"experiment":ep_mod, "exp_50_max": total_exp[0], "exp_99_max": total_exp[1], "exp_max_max": total_exp[2],
                                    "exp_50_true": total_exp[3], "exp_99_true": total_exp[4], "exp_max_true": total_exp[5]}], 
                os.path.join(self.workdir,csv_name + ".csv"))

    def sia(self, exp_name, ep_mod, csv_name, threshes=(0,10,20,50,100)):
        all_eval_res = self.get_eval_res(exp_name)
        all_eval_fil_res = self.get_eval_res(exp_name + "_fil")

        attack_data = []
        attack_data_paired = []
        by_secret = {}
        rank_by_secret = {}
        for al, eval_res in all_eval_res.items():
            res_pos = {}
            res_neg = {}

            if eval_res and eval_res[-1]:
                for secret, loss_and_ranks, ind, toks in eval_res[-1]:
                    secret = tuple(secret)
                    toks = tuple(toks.tolist())
                    rel_loss, rank, base_rank, base_rank_best = loss_and_ranks
                    rank_by_secret.setdefault(secret,[]).append((rank, base_rank, base_rank_best))
                    res_pos[(ind, toks)] = rel_loss, secret
                    attack_data.append((rel_loss,1))

            eval_fil_res = all_eval_fil_res[al] if al in all_eval_fil_res else None
            if eval_fil_res and eval_fil_res[-1]:
                for non_secret, loss_and_ranks, ind, toks in eval_fil_res[-1]:
                    toks = tuple(toks.tolist())
                    rel_loss, rank, base_rank, base_rank_best = loss_and_ranks
                    res_neg[(ind, toks)] = rel_loss, tuple(non_secret)
                    attack_data.append((rel_loss,0))

            inds = set(res_pos.keys()) & set(res_neg.keys())

            res = []
            for ind in inds:
              s = res_pos[ind][0] - res_neg[ind][0]
              secret = res_pos[ind][1]
              by_secret.setdefault(secret, []).append(s)
              res.append(s)
            attack_data_paired += [(x,1) for x in res] + [(-x,0) for x in res]

        ra = round(roc_auc_score([x[1] for x in attack_data], [x[0] for x in attack_data]),2)
        ra_paired = round(roc_auc_score([x[1] for x in attack_data_paired], [x[0] for x in attack_data_paired]),2)
        
        return add2csv([{"experiment":ep_mod, "sia_len_reg": len(attack_data), "sia_len_paired": len(attack_data_paired), "sia_roc_auc": ra, "sia_roc_auc_paired": ra_paired}],
                os.path.join(self.workdir,csv_name + ".csv"))


    def canary_attack(self, acm1, acm2, avg_funcs, canaries, can_ds, neg_per_pos, exp_name, ep_mod,
                        csv_name, field_pref='', overide=False, log_amounts=False, parse_res=True):
        num_funcs = len(avg_funcs)
        skip_calc = False
        res_path = os.path.join(self.workdir, "canary_attack", exp_name + ".pkl")
        print("res_path:", res_path)
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        print(f"{neg_per_pos=}")
        if not overide:
            if os.path.exists(res_path):
                res, lp_res = pickle.load(open(res_path,"rb"))
                skip_calc = True
        if not skip_calc:
            res = {}
            acm1.load_base_model()
            acm1.load_peft_model()
            acm1.load_base_model()
            acm2.load_base_model()
            acm2.load_peft_model()
            acm2.load_base_model()

            lp_res = {f.__name__: [] for f in avg_funcs}
            for i in trange(0,len(can_ds),8):
                if any("submix" in f.__name__ for f in avg_funcs):
                    logits_b, labels_b = acm1.get_logits_and_labels(model_type='base', dataset=can_ds, i=i)
                logits1, labels1 = acm1.get_logits_and_labels(dataset=can_ds, i=i)
                logits2, labels2 = acm2.get_logits_and_labels(dataset=can_ds, i=i)
                for avg_func in avg_funcs:
                    if "submix" in avg_func.__name__:
                        comb_logits = avg_func(logits1, logits2, logits_b)
                    else:
                        comb_logits = acmodel.comb_logits(logits1, logits2, avg_func, relative=True)
                    log_probs = torch.sum(acmodel.get_log_probs(comb_logits, labels1, len(acm1.tokenizer) - 1, reduction='none'), dim=-1).cpu()
                    lp_res[avg_func.__name__] += list(log_probs)
            for fn in lp_res.keys():
                by_context0 = {}
                by_context1 = {}
                for i in range(len(canaries)):
                    can, amount, al, label = canaries[i]
                    if log_amounts and amount != -1:
                        amount = int(round(np.log2(amount)))
                    lp = lp_res[fn][i]
                    if label:
                        by_context1.setdefault((amount, al), []).append(lp)
                    else:
                        by_context0.setdefault((amount, al), []).append(lp)
                func_res = {}
                for key in by_context1.keys():
                    amount, al = key
                    neg_lps = by_context0.get(key,[]) + by_context0.get((-1, al), [])
                    for pos_lp in by_context1[key]:
                        place = len([x for x in neg_lps if x >= pos_lp])
                        sd = np.std(neg_lps)
                        m = np.mean(neg_lps)
                        er = (pos_lp - m) / sd
                        func_res.setdefault(amount, []).append((place, er.item(), al))
                res[fn] = func_res
            pickle.dump([res, lp_res], open(res_path,"wb"))
        if parse_res:
            self.parse_canary_attack_res([res_path], ep_mod, csv_name, field_pref)
        else:
            return res_path

    def parse_canary_attack_res(self, paths, ep_mod, csv_name, field_pref=''):
        res = {}
        for path in paths:
            r,_ = pickle.load(open(path,"rb"))
            for fn, func_res in r.items():
                if fn not in res:
                    res[fn] = {}
                for k, vals in func_res.items():
                    res[fn].setdefault(k,[])
                    res[fn][k] += vals
        data = []
        for fn, func_res in res.items():
            exp_suf = ''
            if len(res) > 1:
                exp_suf = '_' + fn
            record = {"experiment":ep_mod+exp_suf}
            for amount, vals in sorted(func_res.items()):
                ers = [x[1] for x in vals]
                median_er = np.median(ers)
                est_median_exp = -np.log2(scipy.stats.norm(0, 1).cdf(-median_er))
                record[f"{field_pref}attack_score_{amount}"] = est_median_exp
            data.append(record)
        return add2csv(data, os.path.join(self.workdir, csv_name + ".csv"))

    def get_canaries_ds(self, ds, words, can_len=7, amounts=(1,3,10,30,100), seed=1234):
        res = []
        random.seed(seed)
        canaries = []
        for al in self.access_levels:
            for amount in amounts:
                can = [random.choice(words) for _ in range(can_len)]
                can_text = " ".join(can)
                canaries.append((can_text, amount, al))
                row = {k:None for k in self.ds_keys}
                row[self.text_column] = can_text
                row[self.ac_col] = al
                res += [row] * amount
        return canaries, Dataset.from_list(list(ds) + res)

    def add_false_canaries(self, canaries, words, amount_false, replace_range=(2,5), seed=2345):
        new_cans = []
        test_ds = []
        random.seed(seed*2)
        words = words[:]
        random.shuffle(words)
        for can_text, amount, al in canaries:
            can = can_text.split()
            new_cans.append((can_text, amount, al, 1))
            row = {k:None for k in self.ds_keys}
            row[self.text_column] = can_text
            row[self.ac_col] = al
            test_ds.append(row)
            for i in range(amount_false):
                j, k = replace_range
                can[j:k] = [random.choice(words) for _ in range(k-j)]
                can_text = " ".join(can)
                new_cans.append((can_text, amount, al, 0))
                row = {k:None for k in self.ds_keys}
                row[self.text_column] = can_text
                row[self.ac_col] = al
                test_ds.append(row)
        return new_cans, Dataset.from_list(test_ds)
