# MESS Multi-Graph Reconstruction XDP Experiment Design

## 1. Objective

This experiment measures how much XDP leakage a fixed cross-shard inference
method can organize and exploit from the complete server view. Its output is
not an ordinary DP value independent of distance. Instead, it is a function of
the normalized angular distance between the original embeddings:

```text
d_theta -> xi_A(d_theta, delta)
```

Here, `A` denotes Anchor, MDS, Topology, GSGM, or Joint.

## 2. Adjacent Worlds

Fix a real challenge pair `q=(x0,x1)`. The two databases have identical
background records, record counts, and identifiers; they differ only in the
embedding associated with the challenge identifier:

```text
D0 = D_bg union {(id*,x0)}
D1 = D_bg union {(id*,x1)}
```

Each trial independently reruns random `t`-of-`M` routing, storage-side
randomized response, shard-local graph construction, and cross-shard
inference. All method parameters, seed fractions, candidate counts, and
thresholds must be fixed before held-out evaluation.

## 3. Multi-Graph Reconstruction

An inference method receives only the declared target anchors, background seed
correspondences, perturbed binary codes, and shard-local graph structures. It
does not receive the target's complete routing information. Each destination
shard produces one candidate, and candidates are ranked using a fixed
confidence score. Because the routing multiplicity `t` is public, the method
retains the highest-ranked `t-anchor_count` candidates and combines them with
the known anchors to form a fixed-size reconstruction.

Incorrect candidates are not removed using ground truth. Each remains in the
long reconstructed vector as its actual perturbed code. Ground truth is
revealed only after matching to evaluate the number of correctly recovered
target entries and their effective hash distances.

## 4. Long-Vector Distance Score

For reconstructed block `y_s` in shard `s`, use the fixed score:

```text
T_s = Hamming(y_s,H_s(x1)) - Hamming(y_s,H_s(x0)).
```

Summing over all reconstructed blocks gives `T_A,q`. A correctly recovered
target block provides systematic evidence for the corresponding world. If an
incorrect block is independent of the challenge value, its distance to either
template is approximately `kappa/2`, and its expected distance difference is
therefore close to zero. The implementation also reports the actual normalized
distance of incorrect blocks to test this random-`1/2` assumption.

The value `kappa/2` must not be added directly to the XDP distance for an
incorrect block. Doing so would assign the largest privacy budget to a fully
random reconstruction, contradicting its zero effective challenge leakage.

## 5. Linkage-Aware Effective Distance

After a trial ends, ground truth identifies which entries in the final
reconstruction actually belong to the target. For trial `j`, the effective
number of differing coordinates is:

```text
U_A,q^(j) =
  sum over correctly reconstructed shard s
  Hamming(H_s(x0),H_s(x1)).
```

Under the common-null model, incorrect or missing blocks contribute zero to
`U_A,q`. Repeating the experiment in both worlds gives an empirical
distribution of `U_A,q`. For every possible `u`, the implementation constructs
the exact randomized-response privacy-loss distribution, mixes these
distributions using their empirical probabilities, and inverts the result at
the target `delta`.

This calculation is reported in:

```text
linkage_aware_xdp.delta_rows[].xdp_total_budget_xi_point
```

A distribution-free one-sided tolerance limit additionally reports:

```text
Pr[effective pure privacy loss > xi] <= beta
```

together with its confidence level. If the sample size is insufficient, the
status is `insufficient_samples`; the implementation does not fabricate a
finite upper bound.

## 6. End-to-End Two-World XDP

Independent calibration trials fix the bins of the long-vector score.
Held-out trials then estimate:

```text
P_A,q = Law(T_A,q | D0)
Q_A,q = Law(T_A,q | D1).
```

The implementation computes the hockey-stick profile in both directions and
inverts it to obtain a pair-specific total budget for each target `delta`.
Because the score is evaluated after reconstruction, this result retains the
effects of incorrect candidates and data-dependent candidate selection:

```text
reconstructed_vector_xdp.delta_rows[]
```

## 7. Angular-Distance XDP Curve

Every challenge pair retains its actual normalized angular distance:

```text
d_theta(x0,x1) = acos(cosine(x0,x1))/pi.
```

For a specified radius `r`, the implementation aggregates only real pairs
satisfying `d_theta<=r` and takes the maximum pair-specific budget for each
method. The resulting curve is stored in:

```text
reconstructed_angular_xdp_curve.rows[]
```

The row for `r=0.05` and `delta=0.01` can be compared with the
`xi approximately 20.04` reference value obtained under the parameters of the
original LSHRR paper. Reports must also state the flip probability, code length
per shard, routing multiplicity, whether IsoHash is fixed, and the evaluated
challenge-pair range.

## 8. Running the Experiments

Check the dataset paths:

```bash
python3 run.py --dataset sift --check
```

Run the standard SIFT reconstruction experiment:

```bash
bash scripts/run_reconstructed_sift.sh
```

Run the complete profile:

```bash
python3 run.py --dataset sift --profile full --mode all
```

Run all configured datasets:

```bash
bash scripts/run_all_standard.sh
```

## 9. Main Outputs

```text
results/<dataset>/security_results.json
results/<dataset>/security_results.csv
results/<dataset>/reconstructed_multigraph_xdp.csv
results/<dataset>/attack_channel_leakage.csv
results/<dataset>/angular_xdp_curve.csv
```

The paper primarily uses `reconstructed_multigraph_xdp.csv`, which reports:

- the actual angular distance;
- the number of correctly reconstructed target entries;
- the effective IsoHash distance;
- the random-`1/2` check for incorrect blocks;
- linkage-aware RR-PLD XDP;
- two-world long-vector score XDP; and
- XDP envelopes at different angular-distance radii.

## 10. Interpreting the Results

The linkage-aware distance result answers how much fixed-hash information is
exposed by the target entries correctly organized by the inference method.
The end-to-end score result answers how strongly the specified inference
method distinguishes the two adjacent worlds from the complete
reconstruction.

Both are empirical results for the declared dataset, fixed IsoHash mappings,
challenge pairs, auxiliary information, and inference procedure. The
unrestricted mechanism-level guarantee remains the complete-view XDP theorem
proved in the paper.
