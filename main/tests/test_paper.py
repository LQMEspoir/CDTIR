import json
import csv
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from tourism_ie.models.registry import DEFAULT_TARGET_MODULES, resolve_model
from tourism_ie.paper import PRESETS, VARIANTS, preset_snapshot, build_pipeline_inputs, paper_report
from tourism_ie.prompts import system_prompt, RELATION_LABELS
from tourism_ie.data_utils import messages_for, target_text
from tourism_ie.router_multilora import ROUTER_GROUPS, SimpleCategoryRouter, _encode_router_dataset
from tourism_ie.training import swift_token_accuracy, _resume_value


class PaperContractTests(unittest.TestCase):
    def test_router_encoder_never_exposes_gold_answer(self):
        class TinyTokenizer:
            pad_token_id=0; eos_token_id=1
            def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
                return "|".join(x["content"] for x in messages)+("|ASSISTANT" if add_generation_prompt else "")
            def __call__(self,text,add_special_tokens=False):
                return {"input_ids":[ord(ch)+2 for ch in text]}
        row={"input":"原文", "output":[{"entity_text":"绝密答案","entity_label":"目的地"}]}
        # row={"input":"original text", "output":[{"entity_text":"secret answer","entity_label":"Destination"}]}
        encoded=_encode_router_dataset(TinyTokenizer(),"ner",[row],512,"paper_ner","original")[0]
        answer_ids={ord(ch)+2 for ch in "绝密答案"}
        # answer_ids={ord(ch)+2 for ch in "secret answer"}
        self.assertTrue(answer_ids.issubset(set(encoded["input_ids"])))
        self.assertTrue(answer_ids.isdisjoint(set(encoded["router_input_ids"])))
        self.assertLess(len(encoded["router_input_ids"]),len(encoded["input_ids"]))

    def test_four_router_groups(self):
        self.assertEqual(list(ROUTER_GROUPS["ner"]),["location","time","service","attribute"])
        self.assertEqual(set().union(*ROUTER_GROUPS["ner"].values()),set([
            "目的地","出发地","返程地","时长","时间","餐饮","住宿","产品","交通",
            # ["Destination","Origin","Return","Duration","Time","Dining","Lodging","Product","Transport",
            "预算","人数","人群","强度","人气","天气"]))
            # "Budget","People","Crowd","Intensity","Popularity","Weather"]))

    def test_padding_does_not_change_router_logits(self):
        torch.manual_seed(1); router=SimpleCategoryRouter(6,4).eval()
        hidden=torch.randn(1,3,6); padded=torch.cat([hidden,torch.randn(1,2,6)],dim=1)
        a=router(hidden,torch.tensor([[1,1,1]])); b=router(padded,torch.tensor([[1,1,1,0,0]]))
        torch.testing.assert_close(a,b)

    def test_prompts(self):
        eh=system_prompt("relation","paper_re_eh"); to=system_prompt("relation","paper_re_to")
        self.assertNotIn("输入实体",eh)
        # self.assertNotIn("input entities",eh)
        self.assertNotIn("entity_label",to); self.assertNotIn("实体类型组合",to)
        # self.assertNotIn("entity_label",to); self.assertNotIn("entity type combination",to)
        for label in RELATION_LABELS: self.assertIn(label,to)
        msgs=messages_for("relation","北京玩三天","paper_re_e",[
            # messages_for("relation","Beijing three-day trip","paper_re_e",[
            {"entity_text":"北京","entity_label":"目的地"},{"entity_text":"三天","entity_label":"时长"}])
            # {"entity_text":"Beijing","entity_label":"Destination"},{"entity_text":"three days","entity_label":"Duration"}])
        entity_json=msgs[-1]["content"].split("文本对应实体如下：",1)[1]
        # entity_json=msgs[-1]["content"].split("entities for the text are as follows:",1)[1]
        self.assertEqual(len(json.loads(entity_json)),2)
        self.assertTrue(msgs[-1]["content"].startswith("请处理以下文本：北京玩三天"))
        # self.assertTrue(msgs[-1]["content"].startswith("please process the following text: Beijing three-day trip"))

    def test_paper_targets_match_tables(self):
        ner={"output":[{"entity_text":"北京","entity_label":"目的地"},
                       {"entity_text":"三天","entity_label":"时长"}]}
        # ner={"output":[{"entity_text":"Beijing","entity_label":"Destination"},
        #                {"entity_text":"three days","entity_label":"Duration"}]}
        text=target_text("ner",ner,"canonical","paper_ner")
        self.assertEqual(len(text.splitlines()),2)
        for line in text.splitlines(): json.loads(line)
        self.assertEqual(target_text("ner",{"output":[]},"canonical","paper_ner"),"没有找到任何实体")
        # self.assertEqual(target_text("ner",{"output":[]},"canonical","paper_ner"),"no entity found")
        rel={"output":[{"subject":"北京","predicate":"停留时长","object":"三天"}]}
        # rel={"output":[{"subject":"Beijing","predicate":"Stay Duration","object":"three days"}]}
        self.assertEqual(json.loads(target_text("relation",rel,"canonical","paper_re_e"))[0],
                         {"ob1":"北京","rel":"停留时长","ob2":"三天"})
                         # {"ob1":"Beijing","rel":"Stay Duration","ob2":"three days"})
        self.assertEqual(target_text("relation",{"output":[]},"canonical","paper_re_e"),"找不到关系")
        # self.assertEqual(target_text("relation",{"output":[]},"canonical","paper_re_e"),"no relation found")

    def test_swift_shift_and_mask(self):
        labels=np.array([[-100,3,4,-100]])
        pred=np.array([[3,9,8,0]])
        # after shift: predictions positions 0/1 are compared with labels 1/2
        self.assertEqual(swift_token_accuracy(pred,labels),0.5)

    def test_presets_and_targets(self):
        self.assertEqual(tuple(PRESETS),VARIANTS)
        self.assertEqual(DEFAULT_TARGET_MODULES,("q_proj","k_proj","v_proj","o_proj"))
        self.assertEqual(resolve_model("qwen2-1.5b-relation-legacy").target_modules,("q_proj","v_proj"))
        for variant in VARIANTS:
            s=preset_snapshot(variant); self.assertEqual(s["lora_r"],8); self.assertEqual(s["lora_alpha"],32)
            self.assertEqual(s["lora_dropout"],0.1); self.assertEqual(s["seed"],42)

    def test_registered_models_use_parent_models_directory_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            workspace=Path(d)/"CDTIR"; model_dir=Path(d)/"models"/"Qwen2.5-0.5B-Instruct"
            model_dir.mkdir(parents=True); (model_dir/"config.json").write_text("{}",encoding="utf-8")
            old=os.environ.get("CDTIR_ROOT")
            os.environ["CDTIR_ROOT"]=str(workspace)
            try:
                self.assertEqual(resolve_model("qwen2.5-0.5b").model_name,str(model_dir))
            finally:
                if old is None: os.environ.pop("CDTIR_ROOT",None)
                else: os.environ["CDTIR_ROOT"]=old

    def test_explicit_parent_model_path_keeps_canonical_benchmark_key(self):
        spec=resolve_model("../models/Qwen2.5-7B-Instruct")
        self.assertEqual(spec.key,"qwen2.5-7b")
        self.assertEqual(spec.model_name,"../models/Qwen2.5-7B-Instruct")

    def test_pred_pipeline_never_uses_gold_entities(self):
        ner=[{"text_id":"x","prediction":[]}]
        gold=[{"text_id":"x","input":"t","entities":[{"entity_text":"GOLD"}],"output":[]}]
        pred,_=build_pipeline_inputs(ner,gold)
        self.assertEqual(pred[0]["entities"],[])
        gold[0]["entities"]=[{"entity_text":"CHANGED"}]
        pred2,_=build_pipeline_inputs(ner,gold)
        self.assertEqual(pred,pred2)

    def test_report_separates_backbones_ablations_and_gold(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            summary=root/"outputs/backbone_compare/ner/summary.json"; summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps([{
                "model_key":"qwen2.5-7b","strategy":"router_multilora","status":"ok",
                "micro_precision":1,"micro_recall":1,"micro_f1":1,"macro_f1":1,
                "exact_sample_accuracy":1,"token_accuracy":0.9
            }]),encoding="utf-8")
            pipeline=root/"outputs/paper/CDTIR/pipeline"; pipeline.mkdir(parents=True)
            wrong={"gold":[{"subject":"A","predicate":"游玩顺序","object":"B"}],"prediction":[]}
            # wrong={"gold":[{"subject":"A","predicate":"Play Order","object":"B"}],"prediction":[]}
            correct={"gold":wrong["gold"],"prediction":wrong["gold"]}
            for name,row in (("re_e_pred_predictions.jsonl",wrong),("re_e_gold_predictions.jsonl",correct)):
                (pipeline/name).write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
            mirrored=root/"outputs/paper/CDTIR/predictions"; mirrored.mkdir(parents=True)
            (mirrored/"relation.jsonl").write_text(json.dumps(correct,ensure_ascii=False)+"\n",encoding="utf-8")
            report=Path(paper_report(root,Namespace(output_dir=None))["output_dir"])
            with (report/"table_9.csv").open(encoding="utf-8-sig") as f: table11=list(csv.DictReader(f))
            with (report/"table_10.csv").open(encoding="utf-8-sig") as f: table12=list(csv.DictReader(f))
            with (report/"re_e_gold_upper_bound.csv").open(encoding="utf-8-sig") as f: upper=list(csv.DictReader(f))
            self.assertEqual(table11[0]["backbone"],"Qwen2.5-7B")
            self.assertTrue((report/"figure_4.csv").exists())
            main_re=next(x for x in table12 if x["system"]=="CDTIR" and x["task"]=="relation")
            self.assertEqual(float(main_re["micro_f1"]),0.0)
            self.assertEqual({x["system"] for x in upper},{"CDTIR RE-E-Pred","RE-E-Gold upper bound"})

    def test_resume_guard_prevents_step_overrun(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"checkpoint-5"; p.mkdir()
            with self.assertRaisesRegex(ValueError,"already reached"):
                _resume_value("latest",d,5)


if __name__ == "__main__": unittest.main()
