"""Read-only readiness check for a CDTIR reproduction server."""
from __future__ import annotations
import argparse, importlib, json, platform, sys, os
from pathlib import Path
ROOT=Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve(); sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data_end/data_generate"); ap.add_argument("--model",default="qwen2.5-7b"); ap.add_argument("--load-model",action="store_true"); args=ap.parse_args()
    checks=[]; errors=[]
    def check(name,ok,value):
        item={"check":name,"ok":bool(ok),"value":value}; checks.append(item)
        if not ok: errors.append(item)
    check("python",sys.version_info >= (3,10),platform.python_version())
    for name in ("torch","transformers","peft","datasets","accelerate","safetensors"):
        try: m=importlib.import_module(name); check("import:"+name,True,getattr(m,"__version__","available"))
        except Exception as e: check("import:"+name,False,str(e))
    try:
        import torch
        check("cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
        if torch.cuda.is_available(): check("gpu_memory_gb",torch.cuda.get_device_properties(0).total_memory/1024**3>=16,round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
    except Exception as e: check("torch_runtime",False,str(e))
    try:
        from tourism_ie.paper import validate_paper_data
        check("paper_data",True,validate_paper_data(ROOT,args.data_dir,write_reports=False))
    except Exception as e: check("paper_data",False,str(e))
    try:
        from tourism_ie.models import resolve_model
        check("model_registry",True,resolve_model(args.model).to_dict())
        if args.load_model:
            from tourism_ie.modeling import load_tokenizer,load_base_model
            load_tokenizer(args.model); load_base_model(args.model,training=False); check("model_load",True,"loaded")
    except Exception as e: check("model_load",False,str(e))
    result={"ready":not errors,"checks":checks,"errors":errors,"next":"python run.py paper-train --variant CDTIR --task ner --max-steps 5"}
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str)); raise SystemExit(0 if not errors else 1)

if __name__=="__main__": main()
