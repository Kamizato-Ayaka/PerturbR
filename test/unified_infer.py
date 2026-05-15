#!/usr/bin/env python3
import argparse, json, os, re, shutil, subprocess
from pathlib import Path
from datetime import datetime


def now_tag(): return datetime.now().strftime('%Y%m%d_%H%M%S')

def read_jsonl(p):
    out=[]
    with open(p,'r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: out.append(json.loads(l))
    return out

def write_jsonl(p,rows):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def write_json(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def clean_sentence(x):
    toks = x if isinstance(x,list) else str(x or '').replace('\n',' ').split()
    bad={'.',',',';',':','...','....','.....'}
    out=[];seen=set()
    for t in toks:
        t=str(t).strip()
        if not t or t in bad or set(t)=={'.'} or t in seen: continue
        out.append(t);seen.add(t)
    return out

def build_baseline_prompt(r):
    ctrl = clean_sentence(r.get('control_sentence') or r.get('control_expression') or r.get('input_sentence') or '')
    cmeta = r.get('condition_metadata',{}) if isinstance(r.get('condition_metadata'),dict) else {}
    def g(k): return r.get(k,cmeta.get(k,'N/A'))
    system='You are an AI expert in transcriptomic perturbation analysis. Your task is to directly predict the complete perturbed gene expression sentence after a given perturbation. The output should be a single whitespace-separated sequence of gene symbols.'
    user='[Condition]\nCell line: {0}\nPerturbation type: {1}\nPerturbation name: {2}\nSMILES: {3}\nTreatment time: {4}\n\n[Control Expression]\n{5}\n\n[Task]\nPredict the complete perturbed expression sentence.'.format(g('cell_id'),g('perturbation_type'),g('pert_name'),g('canonical_smiles'),g('pert_itime'),' '.join(ctrl))
    return [{'role':'system','content':system},{'role':'user','content':user}]

def normalize_prompt(p):
    if isinstance(p,dict) and 'messages' in p: p=p['messages']
    if isinstance(p,list): return [{'role':str(m['role']),'content':str(m['content'])} for m in p]
    if isinstance(p,str): return [{'role':'user','content':p}]
    raise ValueError('bad prompt')

def infer_rows(records,mode,planner_out=None):
    rows=[]
    for i,r in enumerate(records):
        if mode=='baseline': msg=build_baseline_prompt(r)
        elif mode=='planner': msg=normalize_prompt(r['planner_prompt'])
        elif mode=='isolated-reranker': msg=normalize_prompt(r['reranker_isolated_prompt'])
        elif mode=='reranker': msg=normalize_prompt(r['reranker_prompt'])
        elif mode=='two-stage-reranker':
            pp = planner_out[i]
            prompt = r.get('reranker_prompt_template') or r.get('e2e_reranker_prompt_template') or r.get('reranker_prompt')
            txt = json.dumps(prompt,ensure_ascii=False) if not isinstance(prompt,str) else prompt
            txt = txt.replace('{planner_output}', pp)
            msg = normalize_prompt(txt)
        else: raise ValueError(mode)
        rows.append({'messages':msg})
    return rows

def run_swift(args,val_path,result_path,run_dir):
    cmd=['swift','infer','--model',args.model,'--infer_backend',args.infer_backend,'--model_type','qwen3','--torch_dtype','float16','--enable_thinking','False','--val_dataset',str(Path(val_path).resolve()),'--vllm_gpu_memory_utilization',str(args.vllm_gpu_memory_utilization),'--vllm_max_model_len',str(args.vllm_max_model_len),'--max_new_tokens',str(args.max_new_tokens),'--result_path',str(Path(result_path).resolve())]
    env=os.environ.copy();env['NPROC_PER_NODE']=str(args.nproc_per_node);env['CUDA_VISIBLE_DEVICES']=args.cuda_visible_devices
    (run_dir/'command.sh').write_text(' '.join(cmd)+'\n',encoding='utf-8')
    if args.dry_run: return
    with open(run_dir/'infer.log','w',encoding='utf-8') as f:
        p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    if p.returncode!=0: raise RuntimeError('swift infer failed')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['baseline','planner','reranker','isolated-reranker','two-stage-reranker'])
    ap.add_argument('--test-data',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--planner-result',default='')
    ap.add_argument('--outdir',default='./id_runs'); ap.add_argument('--run-name',default='')
    ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--nproc-per-node',type=int,default=4); ap.add_argument('--cuda-visible-devices',default='0,1,2,3')
    ap.add_argument('--infer-backend',default='vllm'); ap.add_argument('--vllm-gpu-memory-utilization',type=float,default=0.9)
    ap.add_argument('--vllm-max-model-len',type=int,default=8192); ap.add_argument('--max-new-tokens',type=int,default=8192)
    args=ap.parse_args()
    rs=read_jsonl(args.test_data)
    run=Path(args.outdir)/(args.run_name or f"{args.mode}__{now_tag()}")
    if run.exists() and args.overwrite: shutil.rmtree(run)
    run.mkdir(parents=True,exist_ok=True)
    planner_out=[]
    if args.mode=='two-stage-reranker':
        prow=read_jsonl(args.planner_result)
        for r in prow:
            planner_out.append(r.get('response') or r.get('predict') or r.get('generated_text') or '')
    val=infer_rows(rs,args.mode,planner_out)
    write_jsonl(run/'val_dataset.messages.jsonl',val)
    write_json(run/'run_config.json',vars(args))
    run_swift(args,run/'val_dataset.messages.jsonl',run/'swift_result.jsonl',run)

if __name__=='__main__': main()
