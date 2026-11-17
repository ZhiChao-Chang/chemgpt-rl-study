"""Recompute pilot endpoints from saved molecules and upstream primitives."""
import csv
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, stdev

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent
RDLogger.DisableLog("rdApp.*")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod = load_module("upstream_score_modifier", ROOT / "assets/guacamol/guacamol/score_modifier.py")
    fps = load_module("upstream_fingerprints", ROOT / "assets/guacamol/guacamol/utils/fingerprints.py")
    ref = Chem.MolFromSmiles("CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
    fp_ref = fps.get_fingerprint(ref, "AP")
    sims = mod.ClippedScoreModifier(upper_x=.8)
    tpsas = mod.MaxGaussianModifier(mu=90, sigma=10)
    lps = mod.MinGaussianModifier(mu=4, sigma=1)
    count, maxerr = 0, 0.0
    summaries = json.loads((ROOT / "pilot_results/summaries.json").read_text())
    flattened = []
    checks = 0
    for summary in summaries:
        run = ROOT / "pilot_results" / f"{summary['method']}_seed{summary['seed']}"
        discovery = json.loads((run / "discovery.json").read_text())
        evaluation = json.loads((run / "final_policy.json").read_text())
        unique = {}
        for row in discovery + evaluation:
            mol = Chem.MolFromSmiles(row["raw_smiles"])
            valid = mol is not None and mol.GetNumAtoms() > 0
            assert valid == row["valid"]
            checks += 1
            if valid:
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                assert canonical == row["canonical"]
                sim = DataStructs.TanimotoSimilarity(fps.get_fingerprint(mol, "AP"), fp_ref)
                # Use the actual upstream score modifier and fingerprint code,
                # independently of the production scoring helper.
                score = float((sims(sim) * tpsas(rdMolDescriptors.CalcTPSA(mol)) * lps(Crippen.MolLogP(mol))) ** (1/3))
                maxerr = max(maxerr, abs(score - row["fexo_mpo"]))
                assert abs(score - row["fexo_mpo"]) < 1e-12
                assert MurckoScaffold.MurckoScaffoldSmiles(mol=mol) == row["scaffold"]
                count += 1
        for row in discovery:
            if row["valid"]:
                unique.setdefault(row["canonical"], row)
        assert len(unique) == 512 == summary["actual_unique_valid_discovery_scores"]
        assert len(discovery) == summary["processed_attempts"]
        assert len(evaluation) == 128
        assert summary["sampled_attempts"] == len(discovery) + summary["discarded_unscored_final_batch"]
        scorelist = sorted((r["fexo_mpo"] for r in unique.values()), reverse=True)
        assert abs(mean(scorelist[:10]) - summary["discovered"]["top10"]) < 1e-12
        for threshold in [.5, .7, .8]:
            hits = [r for r in unique.values() if r["fexo_mpo"] >= threshold]
            assert len(hits) == summary["discovered"][f"hits_{threshold}"]
            assert len({r["scaffold"] for r in hits if r["scaffold"]}) == summary["discovered"][f"hit_nonempty_scaffolds_{threshold}"]
            checks += 2
        valid_eval = [r for r in evaluation if r["valid"]]
        flat = {"method": summary["method"], "seed": summary["seed"],
                "unique_valid_discovery": len(unique), "discovery_attempts": len(discovery),
                "actual_descriptor_calls_discovery": len(discovery),
                "actual_descriptor_calls_final_policy": len(evaluation),
                "discovery_top10": mean(scorelist[:10]),
                "discovery_hits_05": summary["discovered"]["hits_0.5"],
                "discovery_hit_scaffolds_05": summary["discovered"]["hit_nonempty_scaffolds_0.5"],
                "discovery_hits_07": summary["discovered"]["hits_0.7"],
                "final_policy_mean_all_attempts": mean(r["fexo_mpo"] for r in evaluation),
                "final_policy_mean_valid_only": mean(r["fexo_mpo"] for r in valid_eval),
                "final_policy_valid_fraction": len(valid_eval) / 128,
                "final_policy_hits_05_all_attempts_fraction": sum(r["fexo_mpo"] >= .5 for r in evaluation) / 128,
                "updates": summary["updates"], "trained_trajectories": summary["trained_trajectories"],
                "elapsed_seconds": summary["elapsed_seconds"]}
        flattened.append(flat)
    with (ROOT / "pilot_results/metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(flattened[0])); w.writeheader(); w.writerows(flattened)
    grouped = {}
    for method in ["frozen", "dap", "elite_replay", "scaffold_replay"]:
        rows = [r for r in flattened if r["method"] == method]
        grouped[method] = {k: {"mean": mean(r[k] for r in rows), "sd": stdev(r[k] for r in rows),
                              "values": [r[k] for r in rows]}
                           for k in rows[0] if k not in ["method", "seed"]}
    (ROOT / "pilot_results/aggregated.json").write_text(json.dumps(grouped, indent=2))
    audit = {"status": "PASS", "runs": len(summaries), "molecule_records_checked": checks - 12*6,
             "valid_molecule_score_parity_checks": count, "max_score_difference": maxerr,
             "scope": "saved-record recount + official fingerprint/score modifier parity, not full GuacaMol package certification",
             "no_hypothesis_significance_claim": True,
             "actual_scoring": "properties() is called for each processed/final-policy attempt; duplicates are recomputed, not cached; invalid=0",
             "main_pilot_scorer_invocations": sum(r['discovery_attempts']+128 for r in flattened)}
    (ROOT / "pilot_results/independent_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    for method, metrics in grouped.items():
        print(method, {k: (round(metrics[k]["mean"], 4), round(metrics[k]["sd"],4))
                       for k in ["discovery_top10", "discovery_hits_05", "discovery_hit_scaffolds_05", "final_policy_mean_all_attempts", "final_policy_hits_05_all_attempts_fraction"]})


if __name__ == '__main__':
    main()
