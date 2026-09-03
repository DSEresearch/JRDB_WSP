# Concrete Experiment: WSP Temporal Reasoning on JRDB-Act

## Decision

Anchor the proposal to one falsifiable experiment:

> Given a focal person's relational 3D state and a specified future interval, predict that person's multi-label activity state at the next evaluation time. Train WSP on 1--3-second intervals and test temporal extrapolation on 4--6-second intervals, using the same structured inputs, observation schedules, training protocol, and approximately matched parameter budget as a time-aware GRU and a time-conditioned Transformer.

The dataset is JRDB-Act. The scientific variable is the temporal-reasoning operator. Raw perception, activity/event-time prediction, intention inference, causal discovery, planning, and multi-agent coordination are not separate aims in this experiment.

## Why this is the right anchor

[JRDB](https://jrdb.erc.monash.edu.au/dataset/) was collected from the JackRabbot mobile robot in crowded indoor and outdoor environments. It provides synchronized panoramic RGB and LiDAR, timestamps, calibration, and 2D/3D pedestrian tracks. JRDB-Act adds individual actions and social annotations. The official site reports approximately 60,000 frames (64 minutes), 1.8 million 3D boxes, and 2.8 million individual-action labels. The [JRDB-Act CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ehsanpour_JRDB-Act_A_Large-Scale_Dataset_for_Spatio-Temporal_Action_Social_Group_and_CVPR_2022_paper.html) describes 11 pose, three human-human interaction, and 12 human-object interaction labels.

This instantiates the program officer's motivating problem of high-level activity understanding from a mobile robot's 3D world representation. It also keeps the proposal focused: official 3D tracks serve as fixed outputs of multimodal perception, and the experiment asks whether WSP is a better temporal-reasoning operator over those structured states.

## Demonstrated WSP foundation

The finalized WSP study established a controlled dynamics-prediction foundation. Relative to a time-conditioned Transformer, WSP reduced equal-weight mean test NRMSE by 21.9% while using about 76.6% fewer trainable parameters, and it obtained the lowest sample-mean extrapolation NRMSE in all three evaluated MuJoCo environments. Those results motivate the hypothesis that WSP may help with unequal time intervals; they do not yet demonstrate 3D activity understanding.

The activity code reuses `PersistentWaveLayer` from `src/wsp_review_models.py`. WSP is optimized with ordinary task-loss backpropagation. It is derived from a wave-state parameterization; it does not replace conventional backpropagation with “wave-function backpropagation.”

## Exact input and output

For every focal person at retained observation \(t_i\), the preprocessing script forms a fixed-dimensional target-centered relational state \(x_i\). The temporal model receives:

- the ordered relational states \(x_1,\ldots,x_i\);
- a known outgoing interval \(\Delta t_i>0\), interpreted as the requested prediction interval to the labeled evaluation time; and
- a padding mask.

The output at step \(i\) is a vector of logits for the multi-label activity state at \(t_i+\Delta t_i\). In the prepared data, that target is the annotation at the next retained evaluation time.

This timing convention is important. WSP receives a specified future interval and predicts state at that time. It does not predict when the next sensor report will arrive, and it does not predict the time of the next physical event. The “next retained observation” is how a future labeled query is constructed for evaluation, not a claim that its arrival time is known operationally.

Each window uses only its final valid transition for the primary loss and metric (`target_mask`). Earlier observations provide temporal context. This avoids counting overlapping interior transitions repeatedly.

## Structured 3D relational state

The implemented representation deliberately isolates temporal reasoning from perception learning.

Focal-person features are finite-difference 3D velocity; 3D box length, width, and height; sine and cosine of heading when supplied; occlusion; and a presence indicator.

For each of the \(K\) nearest people at the same evaluation time, the representation adds relative 3D position and velocity, 3D distance, box dimensions, relative-heading sine and cosine, occlusion, and presence.

Neighbors are ordered deterministically by 3D distance and track identifier, then zero-padded. Activity labels and social-group identifiers are never input features. The current implementation does not include object nodes, a learned graph neural network, RGB embeddings, raw LiDAR features, or ego-motion compensation. Those should not be claimed as completed components.

## Canonical data contract

The preparation command consumes a long-form CSV, not raw JRDB JSON. Required columns are:

```text
sequence_id, frame_id, track_id, timestamp, x, y, z, action_id
```

Optional columns are `length`, `width`, `height`, `heading`, and `occlusion`; documented aliases are accepted. Use repeated rows with identical state fields for multiple activity labels on the same `(sequence, frame, track)`.

Raw JRDB conversion must be done separately and audited:

1. Use official JRDB-Act evaluation keyframes and stable official 3D track identifiers.
2. Express all coordinates in one documented frame. If robot-relative tracks are used, address ego motion before interpreting velocity as person motion.
3. Match actions to 3D tracks without ambiguous or silent drops.
4. Produce an explicit sequence-to-split JSON file. The CLI requires it for real data; random splitting is allowed only in synthetic smoke mode.
5. For the planned 26-label experiment, invoke `--expected_classes 26`.
6. For one-second evaluation keyframes, invoke `--expected_interval_seconds 1`. Track gaps may be integer multiples of that interval.

JRDB access requires registration, and the [official registration page](https://jrdb.erc.monash.edu/register/) lists the dataset license. Confirm institutional use and artifact-release requirements. Release code, split files, schedules/configuration, and checksums; do not redistribute JRDB data.

## Split and observation protocol

The split unit is a complete video sequence. No track, window, or frame may cross train, validation, and test.

Use the official JRDB-Act development/test organization and the validation sequences identified by the [official evaluation toolkit](https://github.com/JRDB-dataset/jrdb_toolkit/tree/main/Action%26Social_grouping_eval). Record the resulting 20-training/7-validation/27-test assignment in the required JSON file. If held-out test labels are unavailable, lock the internal development analysis and use the official evaluation process for the final test.

JRDB is regularly sampled; irregularity is induced as a controlled protocol. It must not be described as native irregular sensing. The preprocessor removes actual observations first and then recomputes elapsed time from retained timestamps. Deterministic schedules are embedded in the prepared NPZ files and described in metadata with source/output hashes. Every model receives the same files.

| File | Schedule | Role |
|---|---|---|
| `train_seen.npz` | Retained index gaps 1, 2, or 3 | Training |
| `val_seen.npz` | Retained index gaps 1, 2, or 3 | Model selection |
| `test_seen.npz` | All available observations | Regular reference |
| `test_random_75.npz` | Retain about 75% of observations | Moderate missingness |
| `test_random_50.npz` | Retain about 50% of observations | Severe missingness |
| `test_burst.npz` | Remove a contiguous 2--4-observation burst | Outage-like missingness |
| `test_unseen_gap.npz` | Retained index gaps 4, 5, or 6 | Temporal extrapolation |

The primary analysis additionally filters physical `delta_t` to 4--6 seconds. This prevents natural track gaps or non-unit source timing from silently changing the endpoint.

## Models and fairness controls

The three implemented temporal cores are:

1. `wsp`: persistent WSP cosine/sine carry with content-conditioned wave parameters;
2. `time_gru`: a stacked GRU with the same continuous-time feature map; and
3. `time_transformer`: a causal Transformer with the same continuous-time feature map.

They use the same deterministic relational features, train-only feature normalization, output-head design, target mask, class weights, optimizer family, early-stopping rule, model seeds, and prepared observation schedules. Baseline widths are chosen to match the WSP trainable-parameter count within 10% or the run fails. Parameter matching is a capacity control; it does not imply equal memory use or latency.

Secondary timing controls replace the correct `delta_t` with a within-window constant, a shuffled interval order, or zero. These are sensitivity controls applied to already prepared observations; they do not create irregular sensing.

## Primary hypothesis and endpoint

> Holding structured 3D inputs, data, observation schedules, optimization, and trainable capacity approximately fixed, WSP trained on specified 1--3-step prediction intervals will achieve higher internal multi-label macro average precision than the validation-selected time-aware baseline on specified 4--6-second prediction intervals.

The primary effect is `macro AP(WSP) - macro AP(validation-selected best baseline)`.

The current metric is an internal sklearn-based multi-label macro AP over classes represented in training. Do not call it the official JRDB-Act metric until equivalence to the official evaluator is verified.

The analysis selects the strongest GRU/Transformer baseline by validation macro AP; pairs models on identical sequence, track, source-frame, target-frame, label, and interval arrays; averages the paired model effect across common training seeds; resamples complete video sequences rather than windows; and reports a 95% paired sequence-bootstrap interval.

The hypothesis is labeled supported only if:

1. the WSP improvement is at least 0.02 absolute macro AP;
2. the paired interval's lower bound is above zero;
3. the selected baseline differs from WSP by no more than 10% in trainable parameters;
4. at least three model seeds are present;
5. at least 20 independent test videos are present; and
6. the data are not synthetic.

Synthetic results are always labeled `synthetic_smoke_test_only`, regardless of score.

Secondary descriptive outcomes are macro/micro F1, micro AP, Brier score, expected calibration error, parameter count, and performance under the regular, random-missing, and burst conditions. The present scripts do not measure peak GPU memory or inference latency, so no new activity-task efficiency claim should be made from them.

## Rigorous guarantee scope

The experiment provides procedural and statistical rigor, not a blanket guarantee of intelligence:

- source and prepared-file hashes, code hashes, run fingerprints, fixed splits, and fixed schedules make the comparison auditable;
- stale `--resume` runs are rejected when the data, code, or protocol fingerprint differs;
- the paired sequence bootstrap quantifies sampling variability over the evaluated videos; and
- analytic zero-time and non-expansive properties apply only to the implemented WSP propagation operator under their stated assumptions.

These do not guarantee causal understanding, intention inference, Navy-domain transfer, OOD reliability, or end-to-end perception. A null result, a capacity-dependent effect, or insensitivity to correct timing is evidence against the primary hypothesis and must be reported.

## Synthetic smoke-test summary

<img width="1800" height="1030" alt="primary_paired_effect" src="https://github.com/user-attachments/assets/e662b589-3cb4-4dc3-8f7b-3c94abb60516" />
Figure 1. Primary paired effect. WSP trailed the validation-selected GRU by -0.2301 macro-AP points. Four of five paired seed effects were negative; the only positive effect (+0.0192) remained below the +0.02 success margin. The displayed sequence-bootstrap interval, **[-0.2301, -0.1042]**, is diagnostic only because it is based on two synthetic sequences, not the minimum 20. It must not be described as statistical significance.

<img width="2200" height="1130" alt="condition_performance_calibration" src="https://github.com/user-attachments/assets/2bae1cdd-5d78-4d79-86a1-b26ee68a2aa1" />
Figure 2. Secondary condition results. WSP was strongest under burst missingness and exceeded the Transformer in several schedules, but the validation-selected GRU was stronger in most regimes and decisively stronger on the primary long-gap endpoint. Lower is better for Brier and ECE. The points are five-seed means evaluated on the same two sequences, not five independent datasets.

<img width="2250" height="1220" alt="timing_intervention_effects" src="https://github.com/user-attachments/assets/7c4aa1d4-c717-492c-a4a7-172be5c978f3" />
Figure 3. Secondary timing controls. Zeroing elapsed time affected WSP more strongly than the baselines and reduced its macro AP in every schedule (mean change -0.1193). Constant and shuffled inputs produced mixed effects. Therefore, this control confirms that the implemented WSP uses a timing signal, but it does not establish that WSP exploits exact elapsed-time structure better than the baselines.
