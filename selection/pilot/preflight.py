"""CPU feasibility checks, not a reproduction of PMO or an efficacy comparison.

Use the public PMO REINVENT prior; preserve its 55-token vocabulary and logits.
No chemistry-specific generation mask, no docking, no activity prediction.
"""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets/main/reinvent/data"
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
RDLogger.DisableLog("rdApp.*")
TOKENS = sorted(set((ASSET / "Voc").read_text().split())) + ["EOS", "GO"]
EOS, GO = TOKENS.index("EOS"), TOKENS.index("GO")


class Generator(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.embedding = nn.Embedding(len(TOKENS), 128)
        self.gru = nn.GRU(128, 512, num_layers=3, batch_first=True)
        self.linear = nn.Linear(512, len(TOKENS))
        mapped = {"embedding.weight": weights["embedding.weight"],
                  "linear.weight": weights["linear.weight"],
                  "linear.bias": weights["linear.bias"]}
        for layer in range(3):
            for suffix in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                mapped[f"gru.{suffix}_l{layer}"] = weights[f"gru_{layer + 1}.{suffix}"]
        self.load_state_dict(mapped, strict=True)

    def forward(self, x, h=None):
        y, h = self.gru(self.embedding(x), h)
        return self.linear(y), h

    def likelihood(self, seq):
        start = torch.full((len(seq), 1), GO, dtype=torch.long)
        logits, _ = self(torch.cat([start, seq[:, :-1]], dim=1))
        lp = logits.log_softmax(-1).gather(-1, seq.unsqueeze(-1)).squeeze(-1)
        # Include the first EOS, exclude all tokens after it. Truncated rows use
        # the actual maximum-length trajectory, without pretending EOS occurred.
        ended_before = torch.cat([torch.zeros((len(seq), 1), dtype=torch.long),
                                  (seq[:, :-1] == EOS).cumsum(1)], dim=1)
        return (lp * (ended_before == 0)).sum(1)

    @torch.no_grad()
    def sample(self, n, max_length=140):
        x = torch.full((n, 1), GO, dtype=torch.long)
        h, seq = None, []
        alive = torch.ones(n, dtype=torch.bool)
        logp = torch.zeros(n)
        for _ in range(max_length):
            logits, h = self(x, h)
            lp = logits[:, 0].log_softmax(-1)
            token = torch.multinomial(lp.exp(), 1).squeeze(1)
            logp += lp.gather(1, token[:, None]).squeeze(1) * alive
            seq.append(token)
            alive &= token != EOS
            x = token[:, None]
            if not alive.any():
                break
        return torch.stack(seq, 1), logp, alive


def decode(row):
    chars = []
    for index in row.tolist():
        if index == EOS:
            break
        chars.append(TOKENS[index])
    return "".join(chars).replace("L", "Cl").replace("R", "Br")


FEXO = Chem.MolFromSmiles("CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
FEXO_FP = rdMolDescriptors.GetAtomPairFingerprint(FEXO, maxLength=10)


def properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {"valid": False, "qed": 0.0, "fexo_mpo": 0.0}
    logp, tpsa = Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol)
    sim = DataStructs.TanimotoSimilarity(
        rdMolDescriptors.GetAtomPairFingerprint(mol, maxLength=10), FEXO_FP)
    sim_score = min(sim / .8, 1.0)
    tpsa_score = math.exp(-.5 * (min(tpsa - 90, 0.0) / 10) ** 2)
    logp_score = math.exp(-.5 * (max(logp - 4, 0.0)) ** 2)
    score = (sim_score * tpsa_score * logp_score) ** (1 / 3)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return {"valid": True, "canonical": Chem.MolToSmiles(mol, isomericSmiles=True),
            "scaffold": scaffold, "acyclic": scaffold == "", "mw": Descriptors.MolWt(mol),
            "logp": logp, "tpsa": tpsa, "qed": QED.qed(mol), "fexo_ap": sim,
            "fexo_mpo": score, "fexo_sim_component": sim_score,
            "fexo_tpsa_component": tpsa_score, "fexo_logp_component": logp_score,
            "physchem_window": 250 <= Descriptors.MolWt(mol) <= 450 and 1 <= logp <= 3 and 60 <= tpsa <= 100,
            "fexo_all_plateaus": sim >= .8 and tpsa >= 90 and logp <= 4}


def aggregate(rows):
    valid = [r for r in rows if r["valid"]]
    distinct = {r["canonical"]: r for r in valid}
    scores = sorted([r["fexo_mpo"] for r in distinct.values()], reverse=True)
    return {"attempts": len(rows), "valid": len(valid), "unique": len(distinct),
            "valid_fraction": len(valid) / len(rows),
            "unique_nonempty_scaffolds": len({r["scaffold"] for r in valid if r["scaffold"]}),
            "acyclic_valid": sum(r["acyclic"] for r in valid),
            "qed_mean_valid": float(np.mean([r["qed"] for r in valid])) if valid else None,
            "qed_ge_08_attempts": sum(r["qed"] >= .8 for r in valid),
            "physchem_window_attempts": sum(r["physchem_window"] for r in valid),
            "fexo_mpo_mean_valid": float(np.mean([r["fexo_mpo"] for r in valid])) if valid else None,
            "fexo_mpo_top10_unique": float(np.mean(scores[:10])) if scores else None,
            "fexo_mpo_ge_08_unique": sum(r["fexo_mpo"] >= .8 for r in distinct.values()),
            "fexo_mpo_ge_07_unique": sum(r["fexo_mpo"] >= .7 for r in distinct.values()),
            "fexo_all_plateaus_attempts": sum(r["fexo_all_plateaus"] for r in valid)}


def main():
    t0 = time.monotonic()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    torch.manual_seed(20261104)
    weights = torch.load(ASSET / "Prior.ckpt", map_location="cpu", weights_only=True)
    model = Generator(weights).eval()
    # Compare fused-GRU outputs with the checkpoint's original GRUCell layout.
    cells = nn.ModuleList([nn.GRUCell(128 if i == 0 else 512, 512) for i in range(3)])
    for i, cell in enumerate(cells):
        cell.load_state_dict({k.split(".", 1)[1]: v for k, v in weights.items()
                              if k.startswith(f"gru_{i + 1}.")})
    x = torch.randint(len(TOKENS), (3, 17))
    with torch.no_grad():
        fused, _ = model(x)
        hs = [torch.zeros(3, 512) for _ in range(3)]
        logits = []
        for t in range(x.shape[1]):
            z = model.embedding(x[:, t])
            for i, cell in enumerate(cells):
                hs[i] = cell(z, hs[i]); z = hs[i]
            logits.append(model.linear(z))
        reference = torch.stack(logits, 1)
        diff = float((fused - reference).abs().max())
    assert diff < 2e-5, diff

    rows, trajectories, truncations, discrepancies = [], [], 0, []
    # Seed and sample count fixed before seeing any scores.
    for b in range(8):
        seq, sample_lp, truncated = model.sample(64)
        with torch.no_grad():
            teacher_lp = model.likelihood(seq)
        discrepancies.append(float((sample_lp - teacher_lp).abs().max()))
        truncations += int(truncated.sum())
        for j, row in enumerate(seq):
            s = decode(row)
            rows.append({"sample": len(rows), "raw_smiles": s,
                         "truncated": bool(truncated[j]), **properties(s)})
        trajectories.append(seq)
        print(f"sampled {len(rows)}", flush=True)
    assert max(discrepancies) < 3e-4, discrepancies
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (out / "prior_samples.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # Trajectories retain original sampled representation, not canonical SMILES.
    torch.save(trajectories, out / "prior_trajectories.pt")

    # One genuine DAP gradient update proves trainability only. No performance
    # conclusion: just one batch, one seed, no held-out efficacy comparison.
    batch = trajectories[0][:16]
    reward = torch.tensor([r["fexo_mpo"] for r in rows[:16]])
    with torch.no_grad():
        lp_prior = model.likelihood(batch)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lp = model.likelihood(batch)
    loss = ((lp_prior + 30 * reward - lp) ** 2).mean()
    before = model.linear.weight.detach().clone()
    optimizer.zero_grad(); loss.backward()
    gradnorm = float(nn.utils.clip_grad_norm_(model.parameters(), 10.0))
    optimizer.step()
    change = float((model.linear.weight.detach() - before).abs().max())
    assert math.isfinite(gradnorm) and gradnorm > 0 and change > 0
    summary = {"status": "PASS_FEASIBILITY_ONLY", "device": "cpu", "threads": 2,
               "seed": 20261104, "torch": torch.__version__, "rdkit": rdBase.rdkitVersion,
               "parameters": sum(p.numel() for p in model.parameters()),
               "weights_sha256": hashlib.sha256((ASSET / "Prior.ckpt").read_bytes()).hexdigest(),
               "fused_vs_original_cell_max_logit_error": diff,
               "sample_vs_teacherforce_max_logprob_error": max(discrepancies),
               "max_length": 140, "truncated_attempts": truncations,
               "prior": aggregate(rows),
               "one_step_trainability": {"batch_size": 16, "dap_sigma": 30, "lr": 1e-4,
                    "loss": float(loss.detach()), "unclipped_gradient_norm": gradnorm,
                    "max_output_weight_change": change},
               "elapsed_seconds": time.monotonic() - t0,
               "limits": ["one seed for engineering preflight", "no formal optimizer comparison",
                          "Fexofenadine MPO reimplementation from official definitions; not yet package-parity certified",
                          "all-plateaus and physchem-window are supplementary custom diagnostics, not official benchmark metrics",
                          "QED, calculated logP, TPSA and similarity are computational proxies, not binding or efficacy",
                          "no empirical GPU memory or production-run timing measurement"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
