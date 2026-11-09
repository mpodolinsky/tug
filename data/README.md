# data

Local data, ignored by git. One folder per dataset (`libero10_groot`, `libero10_pi05`, `liberox_groot`):

```
<dataset>/
  manifest.csv   one row per rollout, as published with the Hub dataset named in fp/config.py
  splits/        LIBERO-X only: the held-out task lists (leave_task_out, leave_signature_out, leave_skill_out/<skill>/test.txt)
  features.npz   pooled features: one float32 array (steps, 6144) per episode, keyed by the manifest `dir`,
                 columns [base image | wrist image | language] of 2048 each
  targets.npz    targets of the auxiliary heads: the failure mode of each failed episode, the timeline phrase of every
                 policy step, and CLIP ViT-B/32 text embeddings of the phrases (format in fp/data/targets.py)
  actions.npz    the action chunks the policy sampled at every call: one float32 array (steps, 10, H, 7) per episode, keyed by
                 the manifest `dir` (H = 16 for GR00T, 10 for pi0.5; 7 = position, rotation, gripper); sample 0 is the executed chunk.
                 The input of the action-based baselines (STAC, ActProbe, FailureSpot); format in fp/data/actions.py
  task_embeddings.npz
                 one float32 vector (1024,) per task, the language conditioning of ActProbe, and `_source` (where the vectors come
                 from: ActProbe's shipped LIBERO-10 vectors, or Qwen3-Embedding-0.6B); format in fp/data/task_embeddings.py
```

`python -m cli.download <dataset>` fetches `manifest.csv`, `splits/` (unchanged), the pooled features and the targets, which the Hub datasets hold as `pooled/features.npz` and `pooled/targets.npz`. `features.npz` can also be produced from the raw rollouts of the Hub datasets:

```
python -m cli.prepare <dataset> <folder holding the raw rollouts, one <dir>/rollout.npz per episode>
```

Each modality is averaged over its tokens (the language tokens over the unpadded ones only); see `fp/data/pooling.py`.

`actions.npz` is extracted from the same raw rollouts (only the one array is read from each file, which takes seconds); every episode is checked on the way (sample 0 equals the executed chunk, the replan step and the number of policy calls match, all values finite):

```
python -m cli.prepare_actions <dataset> <folder holding the raw rollouts, one <dir>/rollout.npz per episode>
```

`task_embeddings.npz` holds the vectors ActProbe starts its LSTM from. LIBERO-10 uses the vectors ActProbe ships (Qwen3-Embedding-0.6B was tried and does not reproduce them: cosine 0.6 to 0.75); LIBERO-X, which ActProbe never ran, uses Qwen3-Embedding-0.6B (last token, L2-normalised; needs `pip install -e ".[targets]"` and the model weights). Each dataset is trained separately, so the two sources never meet in one model:

```
python -m cli.embed_tasks libero10_groot --source shipped --shipped-dir <ActProbe's benchmarks/pi0_libero/data>
python -m cli.embed_tasks liberox_groot --source qwen3
```

`targets.npz` is built from the `labels.json` / `labels.npz` of each episode (same folder layout as above; needs `pip install -e ".[targets]"` for the CLIP text encoder):

```
python -m cli.targets <dataset> <folder holding the labels, one <dir>/labels.json per episode>
```

## The real-robot dataset

`real8_groot` is GR00T on a real robot, eight tasks (basket, blocks, closer_drawer, eggplant, mushroom, pineapple_out_of_drawer, push_pineapple, tomato), 399 episodes (145 successes, 254 failures), about 50 Hz and 6 control steps per policy call. It is not on the Hub. Its folder holds `manifest.csv`, `features.npz` and `targets.npz` in the formats above, built from the collection folder by

```
python -m cli.import_real real8_groot <collection folder>      # needs `transformers` and the CLIP weights, for the phrase embeddings
```

- The pooled features are taken from the cache made when the rollouts were collected: the token embeddings were not kept (a `rollout.npz` holds only the actions), so `cli.prepare` cannot pool them again. Every array is checked (the manifest's number of steps, 6144 columns, float32, finite).
- The source manifest has no task name and no condition: the task is read from `dir` (`<task>/ep<NNN>`) and also stands in for the condition, so the stratified split puts every task into every part.
- The failure onset is the manifest's frame divided by that episode's `replan_steps` (5 in 18 episodes, 6 in the rest); the import checks that it equals the `failure_policy_step` of the labels in all 254 failures.
- There is no `actions.npz`, `task_embeddings.npz` or `splits/`. The saved plans (`predicted_action_chunks`, 16 x 8) hold only the executed 5 to 7 actions of each call, the rest is NaN, so no two plans overlap in time (74% of consecutive calls share no step, 25% share one): STAC, STAC-Single, ActProbe and FailureSpot cannot be run, and the baseline commands list them as skipped. The actions are also 8-dimensional (position, quaternion, gripper), not 7.
- Splits: one per task (`loto_<task>`, leave one task out), see `docs/SPLITS.md`.
