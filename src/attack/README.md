# MESS XDP Security Experiments v6.5.0

This is a standalone replacement for the earlier experiment scripts. It
separates quantities that must not be mixed:

1. the original random-LSH angular-XDP result;
2. the formal fixed-IsoHash XDP result for each evaluated pair;
3. an empirical angular-distance envelope for the fixed pretrained IsoHash;
4. exact randomized-response PLD accounting;
5. deterministic MDS, topology, GSGM, and joint linkage experiments;
6. a linkage-aware reconstructed-vector XDP experiment;
7. PRR/IRR access- and search-pattern accounting.

The complete experimental methodology is described in
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).

The reconstructed-vector experiment no longer uses
`correctly_linked_replicas * single_shard_xdp` as its principal result.
Instead, it measures the actual shard-specific fixed-IsoHash distance exposed
by the final reconstruction, retains false matches as null blocks, builds
two-world distance-difference score distributions, and reports an empirical
XDP curve indexed by the original angular distance.

## Installation and paths

Python 3.8 is supported.

```bash
cd src/attack
python3 -m pip install -r requirements.txt
```

The runner automatically searches upward for the repository's Compass-compatible
`data/` directory. Expected defaults include:

```text
data/dataset/sift/base.fvecs
data/dataset/laion1m/100k/laion_base.fvecs
data/dataset/trip_distilbert/passages.fvecs
data/dataset/msmarco_bert/passages.fvecs
data/dataset/sift100m/base.bin
```

The IsoHash model is read as `isohash_weights.bin` from the corresponding
dataset directory. Check before running:

```bash
python3 run.py --dataset sift --check
python3 run.py --dataset all --check
```

If discovery fails, specify the repository data root:

```bash
python3 run.py --dataset sift --check \
  --data-root ../../data
```

## Recommended commands

Quick SIFT validation:

```bash
bash scripts/run_quick_sift.sh
```

Standard complete run:

```bash
python3 run.py --dataset sift --profile standard --mode all
```

All available datasets:

```bash
bash scripts/run_all_standard.sh
```

For the original-paper comparison setting:

```bash
python3 run.py --dataset sift --profile standard --mode xdp \
  --flip-probability 0.21 \
  --target-angular-distances 0.01 0.05 0.10 \
  --angular-tail-probabilities 0.01
```

Available modes are `all`, `xdp`, `query`, `inference`, `attack`, and
`validation`.
Missing datasets are skipped for `--dataset all`; add `--strict-all` to stop
on the first missing dataset.

Run only the matching and two-world attack-channel evaluation with:

```bash
bash scripts/run_reconstructed_sift.sh
```

## What is calculated

### Original random-LSH reference

The deterministic regression reproduces the original LSHRR setting

```text
p=0.21, kappa=128, d_theta=0.05, delta=0.01
```

and must return approximately

```text
angular-XDP xi = 20.0410774234
ordinary LDP epsilon = 169.5904530872
```

This formula implements Proposition 5 of Fernandes, Kawamoto, and Murakami,
*Locality Sensitive Hashing with Extended Differential Privacy* (ESORICS
2021, arXiv:2010.09393). It assumes independent random LSH bits satisfying
`Pr[h(x) != h(x')] = d_theta(x,x')`. It is a reference, not an automatically
valid theorem for one fixed pretrained IsoHash.

### Formal fixed-IsoHash accounting

For every real pair, the code first reports its normalized angular distance
`d_theta=acos(cosine)/pi`, then computes

```text
d_s = Hamming(H_s(x_0), H_s(x_1))
U_R = sum_{s in R} d_s
```

over the actual uniform `t`-of-`M` route. Conditioned on the frozen mappings,
the pure privacy cost is `eta*U_R`, where `eta=log((1-p)/p)`. Dynamic
programming obtains the exact route distribution, while the PLD accountant
also integrates randomized-response output randomness.

This is the fixed-mapping form corresponding to Proposition 2 of the original
paper, followed by multi-replica composition. It is a valid pair-specific
fixed-hash guarantee. Its metric is the
fixed-hash Hamming pseudometric, not the original angular metric.

### Empirical angular-distance envelope for fixed IsoHash

This new experiment makes the angle enter the calculation explicitly. For
each radius `r`, it repeatedly:

1. chooses a real anchor that has a neighbor with `d_theta<=r`;
2. chooses one qualifying real neighbor;
3. draws a uniform route;
4. computes fixed-IsoHash mismatch count `U`;
5. draws the corresponding RR privacy loss
   `L=(2Z-U)eta`, `Z~Binomial(U,1-p)`.

A distribution-free one-sided tolerance limit then reports, simultaneously
over the configured radii and views,

```text
Pr[U > u] <= delta
Pr[L > epsilon] <= delta
```

with confidence at least `1 --angular-confidence-beta` under the explicitly
reported finite-pool pair-and-route distribution. Curves are made monotone in
the angular radius by taking a running maximum.

Two views are always separated:

```text
observed_shards=1   single selected replica (diagnostic comparison)
observed_shards=t   complete t-replica server view (MESS result)
```

The one-replica value may be close to the original paper's value. It must not
be presented as the complete multi-graph budget. The complete-view value
normally grows because the server receives `t` independently perturbed
replicas.

This empirical envelope is not an unrestricted-domain angular-XDP theorem.
It is a high-confidence result for the declared real-pair sampling
distribution. The JSON records this scope explicitly.

The default tail is `0.01`. Smaller tails require many trials. If the sample
size is insufficient, the result is `insufficient_samples` and the numerical
upper bound is `null`. Increase, for example:

```bash
--angular-envelope-trials 10000 \
--angular-tail-probabilities 0.01 0.001
```

## Inference experiments

The linkage experiment constructs noisy shard-local k-NN graphs and evaluates
MDS geometry, topology descriptors, graph-signal descriptors, and a fixed
joint score. No learned classifier is used. Ground-truth IDs are consulted
only after scoring. Hit@1 and MRR measure the tested methods; they are not
relabeled as formal XDP guarantees.

### Linkage-aware reconstructed-vector XDP

For every real challenge pair `(x0,x1)`, the experiment constructs adjacent
worlds with the same background records and identifiers. Only the target
embedding changes. Routing, storage-side RR, shard-local graph construction,
and linkage are rerun in every trial.

The method does **not** receive the complete target route. It receives a
declared number of target anchors, a fixed fraction of background seed
correspondences, all noisy shard-local codes, and the sampled graph
structures. MDS, topology, GSGM, and the fixed joint score return one
confidence-ranked candidate per destination shard. The reconstruction keeps
the highest-ranked `t-anchor_count` candidates, because the replication factor
`t` is public.

No ground-truth filtering is performed. A wrong candidate remains in the
reconstructed vector. If it is independent of the challenge value, its
normalized Hamming distance to either candidate template is approximately
`1/2`. The program measures and reports this null-distance assumption instead
of silently imposing it.

For one selected block `y_s`, the fixed distance-difference score is

```text
T_s = Hamming(y_s,H_s(x1)) - Hamming(y_s,H_s(x0)).
```

The reconstructed long-vector score is the sum over all selected shard
blocks. Correct target blocks give systematic evidence for one world; a
world-independent false block has approximately equal distance to both
templates and therefore contributes mean-zero evidence. This avoids the
incorrect operation of adding `kappa/2` random mismatches to the XDP cost.

Two complementary XDP calculations are produced.

1. **Linkage-aware distance composition.** After matching has finished,
   ground truth is used only for evaluation to identify which selected blocks
   are genuine target blocks. Their actual shard-specific clean-code
   distances are summed:

   ```text
   U_A = sum_{s in correctly reconstructed blocks}
         Hamming(H_s(x0),H_s(x1)).
   ```

   Repeated trials give an empirical distribution of `U_A` in both worlds.
   The exact RR PLD is calculated for each integer value and mixed using this
   empirical distribution. Incorrect blocks contribute zero effective
   coordinates under the common-null model, rather than `kappa/2`.

2. **End-to-end reconstructed-score XDP.** Independent calibration trials fix
   score bins. Held-out world-0 and world-1 long-vector scores produce a
   two-sided hockey-stick profile. This calculation retains wrong candidates
   and captures selection effects of the complete reconstruction.

For every pair and target delta, both calculations return a total budget
`xi(q,delta)`. Pairs are then grouped by their actual normalized angular
distance. For every requested radius `r`, the code reports the maximum
evaluated budget over real pairs satisfying `d_theta(x0,x1)<=r`. Therefore the
main experimental result is a distance-indexed XDP curve, not an ordinary DP
number detached from input distance.

The distance-composition result is an empirical reconstruction model; the
score result is the privacy profile of the specified scalar reconstruction
channel. Neither replaces the unrestricted complete-view theorem. Their
purpose is to quantify the XDP leakage organized by the evaluated concrete
inference methods.

The old `r * single_shard_xdp` composition is retained in JSON only as a
legacy diagnostic so that earlier results remain reproducible. It is not the
principal v6.5 result.

Standard and full profiles intentionally use different audit sizes. They can
be overridden explicitly:

```bash
python3 run.py --dataset sift --profile standard --mode attack \
  --attack-audit-background 128 \
  --attack-audit-pairs 3 \
  --attack-calibration-runs-per-world 30 \
  --attack-test-runs-per-world 300 \
  --attack-score-bins 5 \
  --attack-known-anchors 1 \
  --single-shard-xdp 20 \
  --single-shard-delta 0.01 \
  --attack-exposure-tail-probabilities 0.05 \
  --attack-seed-fraction 0.20 \
  --attack-aggregation-sizes 0 1 2 4 8 15 \
  --attack-confidence-beta 0.05 \
  --attack-audit-angular-radius 0.10 \
  --reconstruction-xdp-angular-radii 0.05 0.10
```

Increasing test trials improves the tolerance limit but also increases
runtime because routing, perturbation, graph construction, and inference are
rerun. The program does not tune parameters to force a target privacy number.
The default attack-channel seed fraction is 20%, independently of the 10%
seed fraction used by the general linkage benchmark.

`--attack-audit-pairs` is interpreted per requested angular radius. The
standard profile uses 20 calibration and 120 held-out trials per world; the
full profile uses 30 and 300. A small tail such as `0.01` needs substantially
more held-out trials and is reported as `insufficient_samples` when the
distribution-free confidence statement cannot be supported.

## Query experiments

Access-pattern accounting uses a memoized permanent randomized report plus
fresh instantaneous reports. Search-pattern accounting compares reuse of one
permanent report with a workload in which another logical query has an
independent permanent report.

## Outputs

Each dataset produces:

```text
results/<dataset>/security_results.json
results/<dataset>/security_results.csv
results/<dataset>/angular_xdp_curve.csv
results/<dataset>/attack_channel_leakage.csv
results/<dataset>/reconstructed_multigraph_xdp.csv
```

`angular_xdp_curve.csv` contains the paper-ready columns: angular radius,
observed shard count, tail probability, simultaneous confidence, status,
single/complete view label, and both the pure-loss and direct privacy-loss
budgets.

`attack_channel_leakage.csv` reports, separately for the anchor, MDS,
topology, GSGM, and joint channels, their AUC, threshold advantage, mean number
of correctly recovered replicas, descriptive epsilon, and confidence
endpoint. It also records the calibration-selected number of additional
candidates. The `anchor` row is a one-known-occurrence baseline and the oracle
complete-view value remains in the analytical output.

`reconstructed_multigraph_xdp.csv` is the principal reconstruction output. It
contains the effective fixed-IsoHash distance distribution, linkage-aware RR
PLD budgets, failed-block null-distance checks, end-to-end reconstructed-score
budgets, and the angular-radius XDP envelope.

Cross-dataset execution status and combined rows are saved to:

```text
results/all_datasets/security_summary.json
results/all_datasets/security_summary.csv
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```
