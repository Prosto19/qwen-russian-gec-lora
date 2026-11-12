#!/usr/bin/env python3
"""Soup take-home verification: archival audit and optional LIVE model re-test.

Requires Python stdlib for audit:
  python verify_soup.py --evidence soup_final_evidence.zip
For Colab live re-test (requires torch, transformers, peft, saved models and llama-server):
  python verify_soup.py --evidence /content/soup_final_evidence.zip --live \
    --adapter /content/soup_qwen_gec_output \
    --merged /content/qwen_gec_merged \
    --gguf /content/qwen_gec_q4.gguf \
    --test /content/test.jsonl \
    --llama-server /root/.soup/llama.cpp/build/bin/llama-server

The archival audit validates recorded evidence, NOT independent model inference.
The live mode re-runs adapter-enabled vs disabled logits, and base/merged/Q4 outputs.
"""
import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


def read_evidence(path, name):
    path = Path(path)
    if path.is_file():
        with zipfile.ZipFile(path) as zf:
            return zf.read(name)
    return (path / name).read_bytes()


def load_json(evidence, name):
    return json.loads(read_evidence(evidence, name))


def audit(evidence, data_dir=None):
    base = load_json(evidence, "baseline_gec.json")
    lora = load_json(evidence, "lora_gec_results.json")
    merged = load_json(evidence, "merged_gec_results.json")
    q4 = load_json(evidence, "q4_gec_results.json")
    f16 = load_json(evidence, "f16_gec_results.json")
    proof = load_json(evidence, "adapter_proof.json")
    data_report = load_json(evidence, "report.json")
    run = load_json(evidence, "soup_qwen_gec_output/soup_run.json")
    assert len(base) == len(lora) == len(merged) == len(q4) == len(f16) == 20
    for rows in zip(base, lora, merged, q4, f16):
        assert len({x['test_index'] for x in rows}) == 1
        assert len({x['input'] for x in rows}) == 1
        assert len({x['expected'] for x in rows}) == 1
        assert rows[0]['predicted'] == rows[1]['base']
        assert rows[1]['lora'] == rows[2]['lora'] == rows[3]['lora'] == rows[4]['lora']
    assert len({r['test_index'] for r in base}) == 20, 'Repeated test index'
    em = lambda s, target: s.strip() == target.strip()
    result = {
        'mode': 'archival_evidence_audit_only',
        'sample_size': 20,
        'base_exact': sum(em(r['predicted'], r['expected']) for r in base),
        'lora_exact': sum(em(r['lora'], r['expected']) for r in lora),
        'merged_exact': sum(em(r['merged'], r['expected']) for r in merged),
        'f16_exact': sum(em(r['f16'], r['expected']) for r in f16),
        'q4_exact': sum(em(r['q4'], r['expected']) for r in q4),
        'base_lora_changed': sum(x['predicted'] != y['lora'] for x, y in zip(base, lora)),
        'lora_merged_same': sum(x['lora'] == y['merged'] for x, y in zip(lora, merged)),
        'merged_f16_same': sum(x['merged'] == y['f16'] for x, y in zip(merged, f16)),
        'f16_q4_same': sum(x['f16'] == y['q4'] for x, y in zip(f16, q4)),
        'adapter_proof_recorded': proof,
        'run_id': run['run_id'],
        'data_report_passed': data_report['verification']['passed'],
        'data_splits_reported': {k: v['rows'] for k,v in data_report['verification']['files'].items()},
        'run_data_hash_matches_report': run['training_data_sha256'] == data_report['verification']['files']['train']['sha256'],
    }
    gpu_csv = read_evidence(evidence, 'gpu_train.csv').decode('utf-8')
    gpu_mib = [int(m.group(1)) for row in gpu_csv.splitlines()[1:]
               if (m := re.search(r'\b(\d+) MiB\b', row))]
    result['nvidia_smi_samples'] = len(gpu_mib)
    result['nvidia_smi_max_mib'] = max(gpu_mib)
    result['nvidia_smi_min_mib'] = min(gpu_mib)
    if data_dir:
        result['dataset_hashes'] = {}
        for name in ('train', 'validation', 'test'):
            path = Path(data_dir) / f'{name}.jsonl'
            with path.open('rb') as fh:
                checksum = hashlib.file_digest(fh, 'sha256').hexdigest()
            expected = data_report['verification']['files'][name]['sha256']
            result['dataset_hashes'][name] = {'actual': checksum, 'expected': expected, 'matches': checksum == expected}
    assert result['run_data_hash_matches_report']
    assert result['lora_merged_same'] == 20
    assert result['adapter_proof_recorded']['nonzero_lora_B_count'] == 56
    return result


def generate(model, tokenizer, prompt, torch, disable=False):
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt'
    ).to(next(model.parameters()).device)
    cm = model.disable_adapter() if disable else contextlib.nullcontext()
    with cm, torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()


def serve_gguf(server_binary, gguf, port):
    log = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    p = subprocess.Popen([str(server_binary), '-m', str(gguf), '-c', '2048',
                          '--host', '127.0.0.1', '--port', str(port),
                          '-ngl', '0', '-t', '2'], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if p.poll() is not None:
            raise RuntimeError('llama-server exited. Log: ' + Path(log.name).read_text()[-2000:])
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as r:
                if r.status == 200:
                    return p, log
        except Exception:
            time.sleep(1)
    raise TimeoutError('llama-server never became ready. Log: ' + log.name)


def live(args, audit_result):
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    assert torch.cuda.is_available(), 'Colab NVIDIA GPU required for this re-test'
    test = [json.loads(line) for line in Path(args.test).read_text(encoding='utf-8').splitlines() if line.strip()]
    original = load_json(args.evidence, 'baseline_gec.json')
    items = [(old['test_index'], test[old['test_index']]) for old in original[:args.count]]
    tok = AutoTokenizer.from_pretrained(str(args.adapter))
    base = AutoModelForCausalLM.from_pretrained(args.base_id, dtype=torch.float16, device_map='cuda')
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    norms = [p.detach().float().norm().item() for n, p in model.named_parameters() if 'lora_B' in n]
    deltas, rows = [], []
    for idx, row in items:
        prompt = row['instruction'] + '\n\n' + row['input']
        ids = tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
                                      add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            enabled = model(**ids).logits[:, -1, :].float()
            with model.disable_adapter():
                disabled = model(**ids).logits[:, -1, :].float()
        deltas.append((enabled-disabled).abs().max().item())
        base_answer = generate(model, tok, prompt, torch, disable=True)
        lora_answer = generate(model, tok, prompt, torch)
        rows.append({'test_index':idx, 'input':row['input'], 'expected':row['output'],
                     'base':base_answer, 'lora':lora_answer})
        print(f'adapter/base {len(rows)}/{len(items)}', flush=True)
    assert norms and min(norms) > 0, 'NO-OP risk: zero/absent LoRA B matrices'
    assert max(deltas) > 1e-3, 'NO-OP risk: adapter-on vs adapter-off identical logits'
    del model, base, tok
    gc.collect(); torch.cuda.empty_cache()
    merged_tok = AutoTokenizer.from_pretrained(str(args.merged))
    merged = AutoModelForCausalLM.from_pretrained(str(args.merged), dtype=torch.float16, device_map='cuda').eval()
    for i, r in enumerate(rows):
        r['merged'] = generate(merged, merged_tok, test[r['test_index']]['instruction']+'\n\n'+r['input'], torch)
        print(f'merged {i+1}/{len(rows)}', flush=True)
    del merged
    gc.collect(); torch.cuda.empty_cache()
    p, log = serve_gguf(args.llama_server, args.gguf, args.port)
    try:
        for i, r in enumerate(rows):
            prompt = test[r['test_index']]['instruction'] + '\n\n' + r['input']
            payload = {'messages':[{'role':'user','content':prompt}], 'temperature':0,
                       'max_tokens':384, 'stream':False}
            request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type':'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=300) as resp:
                answer = json.load(resp)['choices'][0]['message']['content'].strip()
            r['q4'] = answer
            print(f'q4 {i+1}/{len(rows)}', flush=True)
    finally:
        p.terminate()
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired: p.kill()
        log.close()
    exact = lambda a,b: a.strip()==b.strip()
    output = {
        'mode':'live_model_inference',
        'count':len(rows),
        'nonzero_lora_B':sum(n>0 for n in norms),
        'lora_B_total':len(norms),
        'max_adapter_on_off_logit_delta':max(deltas),
        'base_exact':sum(exact(r['base'],r['expected']) for r in rows),
        'lora_exact':sum(exact(r['lora'],r['expected']) for r in rows),
        'merged_exact':sum(exact(r['merged'],r['expected']) for r in rows),
        'q4_exact':sum(exact(r['q4'],r['expected']) for r in rows),
        'lora_merged_agreement':sum(r['lora']==r['merged'] for r in rows),
        'rows':rows,
        'limitations': ['Nonzero adapter weights and output deltas cannot prove good generalization.',
          'Only 20 held-out examples; exact match is brittle to valid paraphrases.',
          'GGUF uses llama.cpp and HF uses transformers; engine differences confound attribution.']
    }
    Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:v for k,v in output.items() if k != 'rows'}


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evidence',required=True,help='evidence ZIP or extracted directory')
    parser.add_argument('--data-dir',help='optional actual dataset directory, for SHA-256 checks')
    parser.add_argument('--live',action='store_true',help='run models rather than only audit archived outputs')
    parser.add_argument('--adapter',default='/content/soup_qwen_gec_output')
    parser.add_argument('--merged',default='/content/qwen_gec_merged')
    parser.add_argument('--gguf',default='/content/qwen_gec_q4.gguf')
    parser.add_argument('--test',default='/content/test.jsonl')
    parser.add_argument('--llama-server',default='/root/.soup/llama.cpp/build/bin/llama-server')
    parser.add_argument('--base-id',default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--port',type=int,default=18082)
    parser.add_argument('--output',default='verification_live_output.json')
    args=parser.parse_args()
    result=audit(args.evidence,args.data_dir)
    print('ARCHIVAL EVIDENCE AUDIT:')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.live:
        print('\nLIVE MODEL VERIFICATION:')
        print(json.dumps(live(args,result),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
