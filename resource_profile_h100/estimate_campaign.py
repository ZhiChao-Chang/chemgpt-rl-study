"""Extrapolate measured H100 timings with explicit novelty-rate scenarios."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

p=argparse.ArgumentParser()
p.add_argument('profile',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
d=json.loads(a.profile.read_text())
assert d['status']=='COMPLETE_PROFILE_ONLY_NOT_FORMAL_PERFORMANCE'
assert d['measurement_label']=='ACTUAL_H100_IF_DEVICE_GATE_PASSES'
assert 'H100' in d['gpu']['name'].upper() and d['gpu']['total_memory_bytes']>=70*1024**3
assert d['arguments']['batch_size']==32
methods={}
for name,g in d['methods'].items():
    rows=g['measured_steps']
    methods[name]={'measured_step_seconds_mean':mean(r['total_step_seconds'] for r in rows),
                   'measured_step_seconds_min':min(r['total_step_seconds'] for r in rows),
                   'measured_step_seconds_max':max(r['total_step_seconds'] for r in rows),
                   'mean_timing_stages':{k:mean(r['timings'][k] for r in rows) for k in rows[0]['timings']},
                   'maximum_pytorch_allocated_gib':g['max_peak_allocated_bytes']/1024**3,
                   'maximum_pytorch_reserved_gib':g['max_peak_reserved_bytes']/1024**3,
                   'mean_batch_max_sequence_length':mean(max(r['samples']['token_lengths_including_eos']) for r in rows),
                   'mean_per_molecule_sequence_length':mean(r['samples']['mean_token_length_including_eos'] for r in rows),
                   'natural_eos_fraction':sum(r['samples']['natural_eos'] for r in rows)/(32*len(rows))}
scenarios=[]
for q in [.95,.5,.2]:
    batches=min(math.ceil(3000/(32*q)),math.ceil(15000/32))
    projected={}
    for name,m in methods.items():
        discovery=batches*m['measured_step_seconds_mean']
        final_sampling=32*(m['mean_timing_stages']['sampling_seconds']+m['mean_timing_stages']['decoding_scoring_seconds'])
        projected[name]={'approximate_discovery_batches':batches,'discovery_minutes':discovery/60,
                         'extra_1024_attempt_evaluation_minutes':final_sampling/60,
                         'per_run_total_minutes':(discovery+final_sampling)/60,
                         'six_runs_minutes':6*(discovery+final_sampling)/60}
    total=sum(x['six_runs_minutes'] for x in projected.values())
    scenarios.append({'effective_new_valid_identity_per_attempt':q,'methods':projected,
                      'eighteen_runs_including_final_sampling_minutes':total,
                      'twofold_timing_margin_minutes':2*total})
result={'status':'SCENARIO_ESTIMATE_FROM_ACTUAL_H100_PROFILE','gpu':d['gpu'],'runtime':d['runtime'],
        'measured_methods':methods,'scenarios':scenarios,'synthetic_stress':d['synthetic_stress'],
        'fixed_counts':{'formal_runs':18,'trained_runs':12,'frozen_runs':6,'unique_valid_query_budget_total':54000,
                        'maximum_discovery_attempts_total':270000,'final_policy_attempts_total':18432,
                        'maximum_optimizer_steps_total':5640,'formal_search_train_hard_cap_hours':9},
        'storage_planning':{'one_published_checkpoint_mb_decimal':70.836865,'outputs_and_selected_weights_budget_gb':5,'project_workspace_budget_gb':10},
        'limits':['These are arithmetic scenarios, not an end-to-end 18-run measurement.',
                  'Novelty rates are assumptions, not sustained rates inferred from four batches.',
                  'Training can alter length, validity and repetition; raw timing tables retain length statistics.',
                  'H100 allocation startup and dependency installation are separate one-off overheads.',
                  'Allocator peaks exclude CUDA context/driver accounting.',
                  'Budget tails use smaller batches; constant-batch approximation is not an exact simulator.']}
a.out.write_text(json.dumps(result,indent=2))
print(json.dumps({'measured_methods':methods,'scenarios':[{'q':s['effective_new_valid_identity_per_attempt'],'total_min':s['eighteen_runs_including_final_sampling_minutes'],'margin2x_min':s['twofold_timing_margin_minutes']} for s in scenarios]},indent=2))
