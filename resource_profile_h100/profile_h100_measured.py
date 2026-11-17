#!/usr/bin/env python3
"""Bounded ChemGPT resource profile; DEV reward only, no model saves or network.

This file does not start resources. Execute only under an independently authorized
profile task. CPU is a tiny functional check and never an H100 measurement.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from importlib import metadata

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
REVISION = '7438a282460b3038e17a27e25b85b1376e9a23e2'
WEIGHT_SHA256 = 'a8a39c5ea1d50c450632e003d21c90f847d815a801934b03a12b02010b74d21b'
GUACAMOL_COMMIT = '60ebe1f6a396f16e08b834dce448e9343d259feb'
PERINDOPRIL = 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC'
BLOCKED = (0, 1, 2, 4)
BOS, EOS, PAD = 2, 3, 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--vendor-dir', type=Path, required=True,
                   help='Existing extracted selfies-1.0.4 wheel; no installation/download')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--max-length', type=int, default=256,
                   help='Maximum generated actions, excluding BOS; truncated trajectories have reward 0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--methods', default='frozen,reinforce,reinforce_kl')
    p.add_argument('--stress-shapes', action='store_true',
                   help='Additional synthetic B32/L128 and B32/L256 teacher-force backward peaks')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--kl-beta', type=float, default=0.01)
    a = p.parse_args()
    a.methods = [x.strip() for x in a.methods.split(',') if x.strip()]
    if not a.methods or len(a.methods) != len(set(a.methods)):
        p.error('--methods must be a nonempty, duplicate-free comma list')
    if any(x not in {'frozen', 'reinforce', 'reinforce_kl'} for x in a.methods):
        p.error('unknown method')
    if min(a.batch_size, a.steps, a.max_length, a.threads) < 1:
        p.error('batch-size/steps/max-length/threads must be positive')
    if a.max_length > 512 or a.batch_size > 32 or a.steps > 8:
        p.error('resource profile limits: batch<=32, length<=512, measured steps<=8')
    if a.batch_size < 2 and a.device == 'cuda' and any(x != 'frozen' for x in a.methods):
        p.error('GPU policy-gradient profile uses N>=2 for LOO baseline')
    if a.device == 'cpu' and (a.batch_size > 2 or a.max_length > 16 or a.steps > 2 or a.stress_shapes):
        p.error('CPU functional smoke requires B<=2, L<=16, steps<=2, no stress-shapes')
    if a.learning_rate <= 0 or a.kl_beta < 0:
        p.error('invalid learning-rate or kl-beta')
    return a


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def timed(times, name, device):
    sync(device)
    start = time.perf_counter()
    yield
    sync(device)
    times[name] = time.perf_counter() - start


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def masked_logprobs(logits):
    logits = logits.float().clone()
    logits[..., list(BLOCKED)] = -torch.inf
    return logits.log_softmax(-1)


def rollout(actor, batch_size, max_length, device):
    """Fresh full-softmax token samples, no filtering/deduplication of trajectories."""
    with torch.no_grad():
        x = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        total_lp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        tokens, masks = [], []
        past = None
        for step in range(max_length):
            output = actor(input_ids=x,
                           attention_mask=torch.ones(batch_size, step + 1, dtype=torch.long, device=device),
                           token_type_ids=torch.zeros_like(x),
                           past_key_values=past, use_cache=True)
            lp = masked_logprobs(output.logits[:, -1])
            next_token = torch.multinomial(lp.exp(), 1).squeeze(1)
            masks.append(alive.clone())
            total_lp += lp.gather(-1, next_token[:, None]).squeeze(-1) * alive
            next_token = torch.where(alive, next_token, torch.full_like(next_token, EOS))
            tokens.append(next_token)
            alive = alive & (next_token != EOS)
            x = next_token[:, None]
            past = output.past_key_values
            if not bool(alive.any()):
                break
        return torch.stack(tokens, 1), torch.stack(masks, 1), ~alive, total_lp


def sequence_logprob(model, seq, mask):
    x = torch.cat([torch.full((seq.shape[0], 1), BOS, dtype=seq.dtype, device=seq.device),
                   seq[:, :-1]], dim=1)
    output = model(input_ids=x, attention_mask=mask.long(),
                   token_type_ids=torch.zeros_like(x), use_cache=False)
    lp = masked_logprobs(output.logits)
    token_lp = lp.gather(-1, seq[..., None]).squeeze(-1)
    return (token_lp * mask).sum(-1)


class DevScorer:
    """GuacaMol v2 perindopril_rings: count ECFP4 similarity, aromatic rings Gaussian.

    Source standard_benchmarks.py::perindopril_rings and
    utils/fingerprints.py::get_ECFP4 at GUACAMOL_COMMIT. No formal scorer imports.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        ref = Chem.MolFromSmiles(PERINDOPRIL)
        if ref is None:
            raise RuntimeError('bad fixed development reference')
        self.reference_fp = AllChem.GetMorganFingerprint(ref, 2)

    def __call__(self, seq, natural):
        rewards, canonical, errors, lengths = [], [], {}, []
        seq_cpu, natural_cpu = seq.detach().cpu().tolist(), natural.detach().cpu().tolist()
        valid_count = valid_eos_count = 0
        for ids, ended in zip(seq_cpu, natural_cpu):
            lengths.append(ids.index(EOS) + 1 if EOS in ids else len(ids))
            body = ids[:ids.index(EOS)] if EOS in ids else ids
            sftext = self.tokenizer.decode(body, skip_special_tokens=True).replace(' ', '')
            score, identity = 0.0, None
            try:
                smiles = selfies.decoder(sftext)
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                valid = mol is not None and mol.GetNumAtoms() > 0 and len(Chem.GetMolFrags(mol)) == 1
                if valid:
                    valid_count += 1
                    identity = Chem.MolToSmiles(mol, isomericSmiles=True)
                    if ended:
                        valid_eos_count += 1
                        sim = float(DataStructs.TanimotoSimilarity(
                            AllChem.GetMorganFingerprint(mol, 2), self.reference_fp))
                        rings = rdMolDescriptors.CalcNumAromaticRings(mol)
                        gaussian = math.exp(-0.5 * ((rings - 2.0) / 0.5) ** 2)
                        score = math.sqrt(sim * gaussian)
            except Exception as exc:
                key = type(exc).__name__
                errors[key] = errors.get(key, 0) + 1
            rewards.append(score)
            canonical.append(identity)
        n = len(rewards)
        return rewards, {
            'attempts': n, 'valid_connected': valid_count, 'valid_natural_eos': valid_eos_count,
            'natural_eos': sum(natural_cpu),
            'unique_valid_connected': len({x for x in canonical if x is not None}),
            'unique_valid_natural_eos': len({x for x, e in zip(canonical, natural_cpu) if x is not None and e}),
            'mean_reward_invalid_or_truncated_zero': sum(rewards) / n,
            'min_reward': min(rewards), 'max_reward': max(rewards),
            'token_lengths_including_eos': lengths,
            'generated_action_count': sum(lengths),
            'mean_token_length_including_eos': sum(lengths) / n,
            'decode_error_counts': errors,
        }


def memory(device):
    if device.type != 'cuda':
        return {'peak_allocated_bytes': None, 'peak_reserved_bytes': None,
                'allocated_bytes': None, 'reserved_bytes': None}
    result = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'allocated_bytes': torch.cuda.memory_allocated(device),
              'reserved_bytes': torch.cuda.memory_reserved(device)}
    if result['peak_reserved_bytes'] > 0.25 * torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError('25 percent GPU allocator memory boundary exceeded')
    return result


def measured_step(actor, reference, optimizer, scorer, args, device, previous_ema):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times = {}
    sync(device)
    start = time.perf_counter()
    with timed(times, 'sampling_seconds', device):
        seq, mask, natural, sample_lp = rollout(actor, args.batch_size, args.max_length, device)
    with timed(times, 'decoding_scoring_seconds', device):
        reward_list, statistics = scorer(seq, natural)
        rewards = torch.tensor(reward_list, dtype=torch.float32, device=device)
    ref_lp = None
    if reference is not None:
        with timed(times, 'reference_forward_seconds', device):
            with torch.no_grad():
                ref_lp = sequence_logprob(reference, seq, mask)
    else:
        times['reference_forward_seconds'] = 0.0
    optimization = None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        with timed(times, 'actor_forward_seconds', device):
            actor_lp = sequence_logprob(actor, seq, mask)
            delta = (actor_lp.detach() - sample_lp).abs().max()
            if not bool(torch.isfinite(delta)) or float(delta) > 5e-3:
                raise RuntimeError(f'sampling/teacher-force mismatch: {float(delta)}')
            seq_kl = actor_lp.detach() - ref_lp if reference is not None else torch.zeros_like(rewards)
            regularized_return = (rewards - args.kl_beta * seq_kl).detach()
            if args.batch_size > 1:
                baseline = ((regularized_return.sum() - regularized_return) / (args.batch_size - 1)).detach()
            else:
                baseline = torch.full_like(regularized_return, previous_ema)
            advantage = (regularized_return - baseline).detach()
            loss = -(advantage * actor_lp).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('non-finite policy-gradient loss')
        with timed(times, 'actor_backward_seconds', device):
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        with timed(times, 'adam_step_seconds', device):
            optimizer.step()
        with timed(times, 'optimizer_statistics_seconds', device):
            optimization = {'loss': float(loss.detach()), 'gradient_norm_before_clip': float(grad_norm),
                            'mean_sample_sequence_kl': float(seq_kl.mean()),
                            'mean_regularized_return': float(regularized_return.mean()),
                            'mean_abs_advantage': float(advantage.abs().mean()),
                            'max_sample_vs_teacherforce_logprob_error': float(delta),
                            'baseline': 'leave-one-out' if args.batch_size > 1 else 'previous-batch EMA',
                            'reference_has_gradient': bool(reference is not None and any(
                                p.grad is not None for p in reference.parameters()))}
            previous_ema = 0.9 * previous_ema + 0.1 * float(regularized_return.mean())
    else:
        for key in ['actor_forward_seconds', 'actor_backward_seconds', 'adam_step_seconds',
                    'optimizer_statistics_seconds']:
            times[key] = 0.0
    sync(device)
    elapsed = time.perf_counter() - start
    return {'timings': times, 'total_step_seconds': elapsed,
            'timed_stage_sum_seconds': sum(times.values()),
            'memory': memory(device), 'samples': statistics, 'optimization': optimization}, previous_ema


def build_group(template, method, args, device):
    actor = copy.deepcopy(template).to(device)
    actor.eval()  # Eval mode keeps autograd; all LM dropout configured zero.
    actor.requires_grad_(method != 'frozen')
    ref = None
    if method == 'reinforce_kl':
        ref = copy.deepcopy(template).to(device).eval().requires_grad_(False)
        if any(a.data_ptr() == b.data_ptr() for a, b in zip(actor.parameters(), ref.parameters())):
            raise RuntimeError('actor/reference share parameter storage')
    opt = None if method == 'frozen' else torch.optim.Adam(
        actor.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return actor, ref, opt


def cleanup(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        sync(device)


def synthetic_stress(template, device):
    results = []
    for b, length in [(32, 128), (32, 256)]:
        actor = copy.deepcopy(template).to(device).eval().requires_grad_(True)
        opt = torch.optim.Adam(actor.parameters(), lr=1e-5)
        # Fixed in-vocabulary chemical-body token IDs. Not generated molecules/rewards.
        seq = torch.full((b, length), 115, dtype=torch.long, device=device)
        seq[:, -1] = EOS
        mask = torch.ones_like(seq, dtype=torch.bool)
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        loss = -sequence_logprob(actor, seq, mask).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        sync(device)
        results.append({'kind': 'synthetic_teacherforce_backward_Adam_NOT_molecular_performance',
                        'batch_size': b, 'sequence_length': length,
                        'seconds': time.perf_counter() - t0,
                        'gradient_norm_before_clip': float(grad_norm), 'memory': memory(device)})
        del actor, opt, seq, mask, loss, grad_norm
        cleanup(device)
    return results


def main():
    args = parse_args()
    report = {'status': 'INITIALIZING', 'scope': 'bounded DEV resource profile; no formal task scoring or saved weights',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model_revision': REVISION, 'methods': {}, 'synthetic_stress': [],
              'measurement_label': 'ACTUAL_H100_IF_DEVICE_GATE_PASSES' if args.device == 'cuda' else 'CPU_FUNCTIONAL_ONLY',
              'no_cross_step_replay': True, 'fresh_batch_updates': 1,
              'warmup': 'one real warmup per method, excluded; model and Adam then reinitialized from original weights',
              'reward': {'name': 'DEV Perindopril MPO', 'guacamol_commit': GUACAMOL_COMMIT,
                         'reference_smiles': PERINDOPRIL, 'fingerprint': 'unhashed count Morgan radius2; useFeatures=False',
                         'aromatic_rings_gaussian': {'mu': 2.0, 'sigma': 0.5},
                         'aggregation': 'sqrt(similarity * gaussian)', 'invalid_or_truncated_reward': 0.0},
              'token_contract': {'bos': BOS, 'eos': EOS, 'pad': PAD, 'blocked': list(BLOCKED),
                                 'temperature': 1.0, 'token_type_ids': 'all zero', 'length_normalization': False},
              'gpu_memory_policy': {'allocator_fraction_hard_limit': 0.24,
                                    'peak_reserved_limit_fraction': 0.25,
                                    'scope': 'PyTorch allocator; CUDA driver/context memory is outside allocator accounting'},
              'script_sha256': sha256_file(Path(__file__))}
    save_json(args.out, report)
    try:
        global torch, selfies, Chem, AllChem, DataStructs, rdMolDescriptors
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        from tokenizers import Tokenizer
        sys.path.insert(0, str(args.vendor_dir.resolve()))
        import selfies
        from rdkit import Chem, DataStructs, RDLogger, rdBase
        from rdkit.Chem import AllChem, rdMolDescriptors
        RDLogger.DisableLog('rdApp.*')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; CPU fallback is forbidden for H100 results')
            props = torch.cuda.get_device_properties(device)
            if 'H100' not in props.name.upper() or props.total_memory < 70 * 1024 ** 3:
                raise RuntimeError(f'Full H100 >=70GiB required, got {props.name}, {props.total_memory} bytes')
            torch.cuda.set_per_process_memory_fraction(0.24, device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report['gpu'] = {'name': props.name, 'total_memory_bytes': props.total_memory,
                             'compute_capability': [props.major, props.minor],
                             'multiprocessor_count': props.multi_processor_count}
        torch.set_float32_matmul_precision('highest')
        sf_origin = Path(selfies.__file__).resolve()
        if not sf_origin.is_relative_to(args.vendor_dir.resolve()):
            raise RuntimeError('SELFIES did not load from supplied vendor directory')
        try:
            distribution_version = metadata.version('selfies')
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError('vendor must retain selfies-1.0.4.dist-info') from exc
        if distribution_version != '1.0.4':
            raise RuntimeError(f'Expected selfies distribution1.0.4, got {distribution_version}')
        weight_path = args.model_dir / 'pytorch_model.bin'
        if sha256_file(weight_path) != WEIGHT_SHA256:
            raise RuntimeError('fixed checkpoint SHA256 mismatch')
        template, info = AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=torch.float32,
            weights_only=True, output_loading_info=True)
        template.eval()
        if info.get('missing_keys') or info.get('mismatched_keys') or any(
                not (k.endswith('.attn.attention.bias') or k.endswith('.attn.attention.masked_bias'))
                for k in info.get('unexpected_keys', [])):
            raise RuntimeError(f'unexpected loading discrepancy: {info}')
        tokenizer = Tokenizer.from_file(str(args.model_dir / 'tokenizer.json'))
        vocab = tokenizer.get_vocab()
        if len(vocab) != 684 or template.config.vocab_size != 684 or any(
                vocab.get(k) != i for k, i in {'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}.items()):
            raise RuntimeError('vocabulary contract mismatch')
        template.config.bos_token_id, template.config.eos_token_id, template.config.pad_token_id = BOS, EOS, PAD
        if any(float(getattr(template.config, k, -1)) != 0 for k in ['attention_dropout', 'embed_dropout', 'resid_dropout']):
            raise RuntimeError('dropout must be zero')
        scorer = DevScorer(tokenizer)
        report['runtime'] = {'python': platform.python_version(), 'torch': torch.__version__,
                             'transformers': transformers.__version__, 'rdkit': rdBase.rdkitVersion,
                             'selfies_distribution': distribution_version,
                             'selfies_module': getattr(selfies, '__version__', 'unknown'),
                             'selfies_origin': str(sf_origin), 'cuda_runtime': torch.version.cuda,
                             'actual_parameter_count': sum(p.numel() for p in template.parameters()),
                             'weight_sha256': WEIGHT_SHA256, 'loading_info': info}
        initial_hash = parameter_hash(template)
        report['original_parameter_hash'] = initial_hash
        report['status'] = 'RUNNING'
        save_json(args.out, report)
        for method in args.methods:
            torch.manual_seed(args.seed)
            actor, ref, opt = build_group(template, method, args, device)
            warmup, _ = measured_step(actor, ref, opt, scorer, args, device, 0.0)
            del actor, ref, opt
            cleanup(device)
            torch.manual_seed(args.seed)
            init_start = time.perf_counter()
            actor, ref, opt = build_group(template, method, args, device)
            sync(device)
            group = {'status': 'RUNNING', 'warmup_excluded': warmup,
                     'initialization_seconds': time.perf_counter() - init_start,
                     'measured_steps': [], 'starts_from_original_parameters': parameter_hash(actor) == initial_hash}
            if not group['starts_from_original_parameters']:
                raise RuntimeError('initial actor hash differs from checkpoint')
            report['methods'][method] = group
            ema = 0.0
            for step in range(args.steps):
                row, ema = measured_step(actor, ref, opt, scorer, args, device, ema)
                row['step'] = step
                group['measured_steps'].append(row)
                save_json(args.out, report)
            if ref is not None:
                group['reference_hash_unchanged'] = parameter_hash(ref) == initial_hash
                if not group['reference_hash_unchanged']:
                    raise RuntimeError('reference parameters changed')
            group['status'] = 'COMPLETE'
            group['mean_measured_step_seconds'] = sum(x['total_step_seconds'] for x in group['measured_steps']) / args.steps
            group['max_peak_allocated_bytes'] = max((x['memory']['peak_allocated_bytes'] or 0) for x in group['measured_steps'])
            group['max_peak_reserved_bytes'] = max((x['memory']['peak_reserved_bytes'] or 0) for x in group['measured_steps'])
            del actor, ref, opt
            cleanup(device)
            save_json(args.out, report)
        if args.stress_shapes:
            report['synthetic_stress'] = synthetic_stress(template, device)
        report['status'] = 'COMPLETE_PROFILE_ONLY_NOT_FORMAL_PERFORMANCE'
        save_json(args.out, report)
        print(str(args.out), flush=True)
        return 0
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        save_json(args.out, report)
        print(f'Profile failed; evidence: {args.out}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
