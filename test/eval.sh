NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
  --model vandijklab/C2S-Scale-Gemma-2-2B \
  --infer_backend vllm \
  --val_dataset /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_ood.jsonl \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 12000 \
  --max_new_tokens 8192



python two-stage-eval.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_ood.jsonl \
  --recall-run-dir /root/dengjie/AI4SCI/PerturbR/test/ood_runs/ins_0513__planner_prompt__20260513_152626 \
  --outdir ./eval_results/recall_eval \
  --topks 100,200 \
  --block-size 125



python two-stage-eval.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_ood.jsonl \
  --rerank-run-dir /root/dengjie/AI4SCI/PerturbR/test/ood_runs/ins_0513__reranker_isolated_prompt__20260513_154802 \
  --outdir ./eval_results/rerank_eval \
  --topks 100,200 \
  --block-size 125


python eval_c2s_sentence_v2.py \
  --test-data /path/to/reranker_test.jsonl \
  --rerank-run-dir /path/to/reranker_run \
  --outdir ./eval_v2_rerank \
  --topks 20,50

python two-stage-eval.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_ood.jsonl \
  --end2end-run-dir /root/dengjie/AI4SCI/PerturbR/test/ood_runs/C2S-Scale-Gemma-2-2B__baseline__all__20260511_184916 \
  --outdir ./eval_results/end2end_eval \
  --topks 100,200 \
  --block-size 125


python swift-infer.py --model /root/dengjie/AI4SCI/Model-Saves/ins_0513/v1-20260513-002543/checkpoint-556 \



python swift-two-stage-infer.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl \
  --planner-model /root/dengjie/AI4SCI/Model-Saves/ins_0513/v1-20260513-002543/checkpoint-556 \
  --reranker-model /root/dengjie/AI4SCI/Model-Saves/ins_0513/v1-20260513-002543/checkpoint-556 \
  --outdir ./id_runs \
  --run-name qwen3_two_stage_e2e \
  --nproc-per-node 4 \
  --cuda-visible-devices 0,1,2,3 \
  --vllm-gpu-memory-utilization 0.9 \
  --vllm-max-model-len 8192 \
  --planner-max-new-tokens 8192 \
  --reranker-max-new-tokens 8192

python eval_tools.py \
  --test-data /root/dengjie/AI4SCI/PP-data/GSE92742-TEST/test_id.jsonl \
  --result-jsonl /root/dengjie/AI4SCI/runs/id_runs/qwen3_two_stage_e2e_rerank_only/e2e_result.jsonl \
  --model-dir /root/dengjie/AI4SCI/Model-Saves/scGPT \
  --out /root/dengjie/AI4SCI/runs/id_runs/qwen3_two_stage_e2e_rerank_only/scgpt_metrics.json \
  --device cuda \
  --batch-size 32 \
  --max-length 1200 \
  --save-emb-prefix /root/dengjie/AI4SCI/PerturbR/test/id_runs/qwen3_two_stage_e2e_rerank_only/scgpt_emb