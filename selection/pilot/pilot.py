"""Small exploratory comparison for course-topic selection, not PMO results.

All training variants share a stationary reward and a prior. Replay changes
which stored trajectories enter the DAP loss; it is not an on-policy gradient.
"""
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preflight import ASSET, EOS, Generator, aggregate, decode, properties

ROOT = Path(__file__).resolve().parent
CONFIG = {"purpose": "exploratory course feasibility, not a published-benchmark reproduction",
          "task": "Fexofenadine MPO, official-definition reimplementation",
          "seeds": [19, 23, 29], "methods": ["frozen", "dap", "elite_replay", "scaffold_replay"],
          "budget_unique_valid": 512, "generation_cap": 1536, "batch_size": 32,
          "lr": 1e-4, "sigma": 60, "replay_size": 16, "archive_capacity": 128,
          "max_length": 140, "frozen_policy_evaluation_attempts": 128,
          "hit_thresholds": [.5, .7, .8],
          "scope": "CPU only; no new dependencies, GPU, docking, QSAR, synthesis or server jobs",
          "decisions": ["configuration fixed before comparative pilot",
                        "memory stores raw stationary MPO, never history-dependent novelty rewards",
                        "elite and scaffold replay use equal memory capacity/replay size/update count per batch",
                        "no-replay compute differs: report training trajectory count, no compute-matched superiority claim",
                        "main discovery budget excludes fixed final-policy evaluation, reported separately",
                        "scaffold grouping is an archive heuristic, not biological diversity truth",
                        "final-policy sampling has a separate fixed attempt budget; scoring recorded and charged separately",
                        "do not choose hyperparameters or formal test tasks by pilot winner"]}


def trajectory(row):
    ids = row.tolist()
    if EOS in ids:
        ids = ids[:ids.index(EOS) + 1]
    return ids


def pad(seqs):
    out = torch.full((len(seqs), max(map(len, seqs))), EOS, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = torch.tensor(seq)
    return out


def archive(history, method):
    best = sorted(history.values(), key=lambda r: (-r["fexo_mpo"], r["canonical"]))
    if method == "elite_replay":
        return best[:CONFIG["archive_capacity"]]
    groups = defaultdict(list)
    for row in best:
        groups[row["scaffold"]].append(row)
    # Round robin across scaffold buckets, score-sorted within each bucket.
    groups = sorted(groups.values(), key=lambda g: (-g[0]["fexo_mpo"], g[0]["canonical"]))
    result, level = [], 0
    while len(result) < CONFIG["archive_capacity"]:
        added = 0
        for group in groups:
            if level < len(group):
                result.append(group[level]); added += 1
                if len(result) == CONFIG["archive_capacity"]:
                    break
        if not added:
            break
        level += 1
    return result


def discovered_metrics(history):
    rows = list(history.values())
    scores = sorted((r["fexo_mpo"] for r in rows), reverse=True)
    result = {"unique_valid": len(rows), "top10": float(np.mean(scores[:10])) if scores else 0.0,
              "top1": max(scores) if scores else 0.0}
    for threshold in CONFIG["hit_thresholds"]:
        hits = [r for r in rows if r["fexo_mpo"] >= threshold]
        result[f"hits_{threshold}"] = len(hits)
        result[f"hit_nonempty_scaffolds_{threshold}"] = len({r["scaffold"] for r in hits if r["scaffold"]})
    return result


def run(method, seed, weights, prior):
    t0 = time.monotonic()
    agent = Generator(weights).eval()
    optimizer = torch.optim.Adam(agent.parameters(), lr=CONFIG["lr"])
    torch.manual_seed(seed)
    replay_rng = random.Random(seed)
    history, log, all_attempts = {}, [], []
    processed, sampled, discarded, truncated = 0, 0, 0, 0
    updates, trained_trajectories = 0, 0
    while len(history) < CONFIG["budget_unique_valid"] and sampled < CONFIG["generation_cap"]:
        n = min(CONFIG["batch_size"], CONFIG["generation_cap"] - sampled)
        agent.eval()
        seq, _, trunc = agent.sample(n, CONFIG["max_length"])
        sampled += n
        fresh_seq, fresh_rewards = [], []
        additions = []
        for j, row in enumerate(seq):
            if len(history) == CONFIG["budget_unique_valid"]:
                discarded += 1
                continue
            s = decode(row)
            prop = properties(s)
            processed += 1
            truncated += int(trunc[j])
            ids = trajectory(row)
            fresh_seq.append(ids)
            fresh_rewards.append(prop["fexo_mpo"])
            record = {"raw_smiles": s, "attempt": processed, "truncated": bool(trunc[j]), **prop}
            all_attempts.append(record)
            if prop["valid"] and prop["canonical"] not in history:
                entry = {"raw_smiles": s, "tokens": ids, **prop}
                history[prop["canonical"]] = entry
                additions.append(prop["canonical"])
        if len(history) == CONFIG["budget_unique_valid"]:
            # Do not update after the final discovery batch: no selective
            # early stop/checkpoint is used; last consumed state is fixed.
            log.append({"updates": updates, "attempts": processed, **discovered_metrics(history)})
            break
        if method != "frozen":
            if method.endswith("replay"):
                # Exclude current additions so replay always uses past batches.
                old = {k: v for k, v in history.items() if k not in additions}
                pool = archive(old, method)
                if len(pool) >= CONFIG["replay_size"]:
                    replay = replay_rng.sample(pool, CONFIG["replay_size"])
                    fresh_seq.extend(r["tokens"] for r in replay)
                    fresh_rewards.extend(r["fexo_mpo"] for r in replay)
            x = pad(fresh_seq)
            reward = torch.tensor(fresh_rewards)
            with torch.no_grad():
                lp_prior = prior.likelihood(x)
            agent.train()
            lp = agent.likelihood(x)
            loss = ((lp_prior + CONFIG["sigma"] * reward - lp) ** 2).mean()
            optimizer.zero_grad(); loss.backward()
            norm = nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
            assert torch.isfinite(norm), (method, seed, updates)
            optimizer.step()
            updates += 1
            trained_trajectories += len(x)
        log.append({"updates": updates, "attempts": processed, **discovered_metrics(history)})
    agent.eval()
    # Independent fixed-size final-policy readout, never used for training.
    torch.manual_seed(seed + 100000)
    eval_rows = []
    with torch.no_grad():
        for _ in range(CONFIG["frozen_policy_evaluation_attempts"] // 32):
            seq, _, trunc = agent.sample(32, CONFIG["max_length"])
            for j, row in enumerate(seq):
                s = decode(row)
                eval_rows.append({"raw_smiles": s, "truncated": bool(trunc[j]), **properties(s)})
    final_policy = aggregate(eval_rows)
    final_policy["fexo_mpo_ge_05_attempts"] = sum(r["fexo_mpo"] >= .5 for r in eval_rows)
    final_policy["fexo_mpo_ge_07_attempts"] = sum(r["fexo_mpo"] >= .7 for r in eval_rows)
    final_policy["unique_new_valid_scored"] = len({r["canonical"] for r in eval_rows
        if r["valid"] and r["canonical"] not in history})
    summary = {"method": method, "seed": seed, "status": "budget_reached" if len(history) == CONFIG["budget_unique_valid"] else "attempt_cap",
               "processed_attempts": processed, "sampled_attempts": sampled,
               "discarded_unscored_final_batch": discarded, "truncated_processed": truncated,
               "actual_unique_valid_discovery_scores": len(history),
               "actual_invalid_attempts_scored_as_zero": sum(not r["valid"] for r in all_attempts),
               "updates": updates, "trained_trajectories": trained_trajectories,
               "discovered": discovered_metrics(history), "final_policy": final_policy,
               "elapsed_seconds": time.monotonic() - t0}
    out = ROOT / "pilot_results" / f"{method}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("discovery.json", all_attempts), ("final_policy.json", eval_rows),
                       ("curves.json", log), ("summary.json", summary)]:
        (out / name).write_text(json.dumps(data, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    out = ROOT / "pilot_results"
    out.mkdir(exist_ok=True)
    p = out / "protocol.json"
    if p.exists():
        raise RuntimeError("Pilot protocol exists; do not overwrite/restart silently")
    p.write_text(json.dumps(CONFIG, indent=2))
    weights = torch.load(ASSET / "Prior.ckpt", map_location="cpu", weights_only=True)
    prior = Generator(weights).eval()
    for parameter in prior.parameters():
        parameter.requires_grad = False
    summaries = []
    for seed in CONFIG["seeds"]:
        for method in CONFIG["methods"]:
            summaries.append(run(method, seed, weights, prior))
    (out / "summaries.json").write_text(json.dumps(summaries, indent=2))
    print("PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
