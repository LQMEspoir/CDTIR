# File: scripts/strict_alignment_selftest.py
# Module responsibility: strict sequence-alignment self-test, focusing on prompt-mask / labels / target-token alignment in the relation legacy fixed route.
# Main data flow: build minimal samples -> encode prompt/target -> check -100 mask and supervised-token alignment -> output self-test result.
# Reading guide: read the public functions/classes first, then follow imports to data_utils, modeling, evaluation, and other common modules.
# Maintenance note: this version only adds explanatory comments; it does not modify expressions, control flow, parameter defaults, or call relationships.

"""No-model self-test for strict legacy-alignment invariants."""
from __future__ import annotations
import ast,json,sys,os
from pathlib import Path
ROOT=Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve()
SRC=Path(__file__).resolve().parents[1]/'src'
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from tourism_ie.prompts import NER_ORIGINAL_STRICT_SYSTEM_PROMPT
from tourism_ie.data_utils import render_prompt,target_text,messages_for,read_jsonl
from tourism_ie.relation_legacy import encode_fixed_row,preprocess_spo_entry
from tourism_ie.models import resolve_model
from tourism_ie.strategies import STRATEGIES

def main():
    """[Function] main
    - Responsibility: script entry: parse args, run required validation, and invoke the corresponding business flow per command.
    - Parameters: no explicit business params (possibly self/cls).
    - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
    - Notes: may read/write disk files/model weights; ensure the output directory is writable;inference/tensor paths must keep device and dtype consistent;includes explicit parameter/data validation; raises on invalid input;emits logs or warnings.
    """
    legacy=(ROOT/'legacy/original_scripts/entity_router_rola.py').read_text(encoding='utf-8')
    tree=ast.parse(legacy); vals=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='system_prompt' for x in n.targets) and isinstance(n.value,ast.Constant):
            vals.append(n.value.value)
    assert vals and vals[0]==NER_ORIGINAL_STRICT_SYSTEM_PROMPT
    frame=render_prompt(None,'ner','文本:北京','original_strict',True)
    # frame=render_prompt(None,'ner','text: Beijing','original_strict',True)
    expected=f'<|im_start|>system\n{NER_ORIGINAL_STRICT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n文本:北京<|im_end|>\n<|im_start|>assistant\n'
    # expected=f'<|im_start|>system\n{NER_ORIGINAL_STRICT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\ntext: Beijing<|im_end|>\n<|im_start|>assistant\n'
    assert frame==expected
    orig=json.loads(next((ROOT/'data/original_reference/entity_output_original.jsonl').open(encoding='utf-8')))
    canon=json.loads(next((ROOT/'data/ner/entity_train.jsonl').open(encoding='utf-8')))
    assert target_text('ner',canon,'original')==orig['output']

    # Strict tourism RE: reproduce qwen-t/qw_e notebook user messages exactly.
    import re
    def notebook_prompt(name):
        """[Function] main.notebook_prompt
        - Responsibility: runs the local "notebook prompt" step; extracted to reuse logic, centralize validation and reduce main-flow complexity.
        - Parameters: name.
        - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        - Notes: includes explicit parameter/data validation; raises on invalid input.
        """
        nb=json.loads((ROOT/'legacy/original_scripts'/name).read_text(encoding='utf-8'))
        for cell in nb['cells']:
            src=''.join(cell.get('source',[]))
            m=re.search(r'prompt\s*=\s*"""(.*?)"""\.strip\(\)',src,re.S)
            if m: return m.group(1).strip()
        raise AssertionError(f'prompt not found in {name}')
    rel0=json.loads(next((ROOT/'data/original_reference/relationship_output_type_original.jsonl').open(encoding='utf-8')))
    entity0=json.loads(next((ROOT/'data/supplementary/relationship_output_entity.jsonl').open(encoding='utf-8')))
    text_prompt=messages_for('relation',rel0['input'],'text_only_strict')
    assert len(text_prompt)==1 and text_prompt[0]['role']=='user'
    assert text_prompt[0]['content']==notebook_prompt('qwen-t.ipynb').replace('{text}',rel0['input'])
    assert '文本:文本:' not in text_prompt[0]['content']
    # assert 'text: text:' not in text_prompt[0]['content']
    base,entity_raw=entity0['input'].split('\n实体:',1)
    # base,entity_raw=entity0['input'].split('\nentities:',1)
    dec=json.JSONDecoder(); pos=0; entities=[]
    while pos<len(entity_raw):
        obj,end=dec.raw_decode(entity_raw,pos); entities.append(obj); pos=end
    entity_prompt=messages_for('relation',base,'entity_aware_strict',entities)
    assert len(entity_prompt)==1 and entity_prompt[0]['role']=='user'
    assert entity_prompt[0]['content']==notebook_prompt('qw_e.ipynb').replace('{text}',entity0['input'])
    strict_target=target_text('relation',rel0,'canonical','text_only_strict')
    expected_target='```json\n'+json.dumps([{'ob1':x['subject'],'rel':x['predicate'],'ob2':x['object']} for x in rel0['output']],ensure_ascii=False,indent=2)+'\n```'
    assert strict_target==expected_target
    e_train=read_jsonl(ROOT/'data/relation/relation_train_e.jsonl')
    e_valid=read_jsonl(ROOT/'data/relation/relation_valid_e.jsonl')
    assert len(e_train)==319 and len(e_valid)==80 and all(r.get('entities') for r in e_train+e_valid)
    from tourism_ie.paper import preset_snapshot
    paper=preset_snapshot('CDTIR')
    assert paper['warmup_ratio']==0.01 and paper['save_strategy']=='steps' and paper['save_steps']==20
    assert paper['ner_prompt']=='paper_ner' and paper['router_loss_weight']==0.5
    spec=resolve_model('qwen2-1.5b-relation-legacy')
    assert spec.model_name=='Qwen/Qwen2-1.5B-Instruct'
    assert spec.target_modules==('q_proj','v_proj')
    assert 'relation_legacy_fixed' in STRATEGIES
    class Tok:
        """[Class] Tok
        - Responsibility: encapsulates Tok state and operations so it can be invoked as a standalone component by train/inference/eval flows.
        - Inherits: no explicit business base class.
        - Usage: instantiated then invoked by the training/inference main flow via public methods; internal state is determined by __init__ and saved config.
        """
        eos_token_id=99
        # [Function] main.Tok.__call__
        # - Responsibility: pads variable-length training samples into batch tensors and stacks the router multi-label supervision vector.
        # - Parameters: text,add_special_tokens=False.
        # - Returns: returns the processed object/metrics/tensor/path; structure depends on each return branch.
        # - Notes: mainly in-memory; no side effects beyond the called object itself.
        def __call__(self,text,add_special_tokens=False): return {'input_ids':[ord(c)%50+1 for c in text]}
    enc=encode_fixed_row(Tok(),{'input_text':'PROMPT','target_text':'<a,b,c>'},32)
    first=next(i for i,v in enumerate(enc['labels']) if v!=-100)
    assert first==6 and all(v==-100 for v in enc['labels'][:6])
    x=preprocess_spo_entry({'text':'轴承外圈磨损会导致振动增大。','spo_list':[{'subject':[0,6],'predicate':'导致','object':[9,13]}]})
    # x=preprocess_spo_entry({'text':'Bearing outer-ring wear causes increased vibration.','spo_list':[{'subject':[0,6],'predicate':'causes','object':[9,13]}]})
    assert x['target_text']=='<轴承外圈磨损, 导致, 振动增大>'
    # assert x['target_text']=='<Bearing outer-ring wear, causes, increased vibration>'
    print('STRICT_PROMPT_RUNTIME_MATCH=PASS')
    print('ORIGINAL_TARGET_SAMPLE_MATCH=PASS')
    print('STRICT_NER_LAUNCHERS=PASS')
    print('STRICT_TEXT_ONLY_RE_PROMPT_TARGET=PASS')
    print('STRICT_ENTITY_AWARE_RE_PROMPT_DATA=PASS')
    print('LEGACY_RELATION_REGISTRY=PASS')
    print('RELATION_CAUSAL_LABEL_ALIGNMENT=PASS')
    print('RELATION_SPO_OFFSET_CONVERSION=PASS')
    print('RESULT: PASS')
if __name__=='__main__': main()
