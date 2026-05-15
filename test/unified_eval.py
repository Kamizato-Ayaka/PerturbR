#!/usr/bin/env python3
import argparse, json, re, subprocess
from pathlib import Path
from difflib import SequenceMatcher

def read_jsonl(p):
    o=[]
    with open(p,'r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: o.append(json.loads(l))
    return o

def write_json(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def parse_seq(x):
    s=str(x or '')
    s=re.sub(r'<think>.*?</think>',' ',s,flags=re.S)
    s=re.sub(r'\bBlock\s*\d+\s*:',' ',s,flags=re.I)
    return [t for t in re.split(r'[\s,\[\]\(\){}]+',s) if t]

def get_pred(r):
    for k in ['reranker_output','final_output','response','predict','prediction','generated_text','output','text','answer']:
        if k in r and r[k] not in (None,''): return r[k]
    return ''

def metric(pred,true):
    ps,ts=parse_seq(pred),parse_seq(true)
    pset,tset=set(ps),set(ts)
    jac=len(pset&tset)/max(len(pset|tset),1)
    ratio=SequenceMatcher(None,ps,ts).ratio() if ps and ts else 0.0
    return {'jaccard':jac,'seq_ratio':ratio,'pred_len':len(ps),'true_len':len(ts)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--test-data',required=True)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--mode',required=True,choices=['baseline','planner','reranker','isolated-reranker','two-stage-reranker'])
    ap.add_argument('--output',default='eval_summary.json')
    ap.add_argument('--with-embedding',action='store_true')
    args=ap.parse_args()

    t=read_jsonl(args.test_data)
    rr=read_jsonl(Path(args.run_dir)/'swift_result.jsonl')
    n=min(len(t),len(rr)); details=[]
    for i in range(n):
        gt = t[i].get('gt_sequence','') if args.mode!='planner' else t[i].get('gt_planner_blocks','')
        m = metric(get_pred(rr[i]),gt)
        m.update({'idx':i,'record_id':t[i].get('record_id',i)})
        details.append(m)
    agg={k:sum(d[k] for d in details)/max(len(details),1) for k in ['jaccard','seq_ratio','pred_len','true_len']}
    out={'mode':args.mode,'num_samples':n,'metrics':agg}
    write_json(Path(args.run_dir)/args.output,out)
    write_json(Path(args.run_dir)/'eval_details.json',details)

    if args.with_embedding and args.mode!='planner':
        cmd=['python','test/eval_tools.py','--test-data',args.test_data,'--result-jsonl',str(Path(args.run_dir)/'swift_result.jsonl')]
        subprocess.run(cmd,check=False)

if __name__=='__main__': main()
