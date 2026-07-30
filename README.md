# MESS

MESS is a research prototype for privacy-aware semantic retrieval over
perturbed binary embeddings. It combines:

- IsoHash encoding of high-dimensional embeddings;
- locality-sensitive hashing randomized response (LSHRR);
- a multi-graph HNSW index;
- permanent and instantaneous query perturbation;
- AES-256-GCM encrypted candidate payloads; and
- client-side decryption, duplicate removal, and exact reranking.

The server searches perturbed binary codes with Hamming distance. It does not
perform homomorphic similarity evaluation or ORAM-protected graph traversal.
The client receives one encrypted candidate response and reranks the recovered
embeddings using the original dataset metric.

This repository is an academic artifact, not a production security system.

## Contents

| Path | Purpose |
|---|---|
| `src/serverorigin.cpp` | MESS server, multi-graph construction, search, insertion, and deletion |
| `src/clientorigin.cpp` | Query encoding and perturbation, request generation, decryption, reranking, and evaluation |
| `src/examples/hnswlib_server.cpp` | Plaintext HNSW baseline server |
| `src/examples/hnswlib_client.cpp` | Plaintext HNSW baseline client and evaluator |
| `src/examples/update_example.cpp` | Example insertion and deletion client |
| `src/util/LSH/` | IsoHash training and inference |
| `src/attack/` | XDP accounting and cross-shard linkage experiments |
| `data/download_dataset.py` | Dataset downloader with size checking, resume support, and IPv4 fallback |
| `initialize_isohash.py` | Trains the IsoHash weight files for all datasets |

The CMake build produces:

```text
build/server_origin
build/client_origin
build/hnswlib_server
build/hnswlib_client
build/update_example
```

## Requirements

The C++ implementation requires:

- Linux;
- CMake 3.26 or newer;
- a C++17 compiler;
- OpenMP;
- POSIX threads and sockets; and
- OpenSSL development headers.

On Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake libssl-dev libomp-dev unzip
```

IsoHash initialization requires Python 3 and NumPy:

```bash
python3 -m pip install "numpy>=1.21,<2.0"
```

The security experiments additionally require SciPy:

```bash
python3 -m pip install -r src/attack/requirements.txt
```

Python 3.8 is supported by the attack-analysis package.

## Build

Run the following commands from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To rebuild one executable:

```bash
cmake --build build --target server_origin -j
cmake --build build --target client_origin -j
```

## Dataset Setup

The four main datasets use the same directory organization and filenames as
the public [Compass artifact](https://github.com/Clive2312/compass).
SIFT100M uses the files from
[Nanvivi/SIFT100M-DiskANN](https://huggingface.co/datasets/Nanvivi/SIFT100M-DiskANN).

From the repository root, download all required files with:

```bash
python3 data/download_dataset.py
```

Download only the four Compass datasets:

```bash
python3 data/download_dataset.py --datasets compass
```

Download only SIFT100M:

```bash
python3 data/download_dataset.py --datasets sift100m
```

If the host has no working IPv6 route to Hugging Face:

```bash
python3 data/download_dataset.py --datasets sift100m --ipv4
```

The downloader:

- skips files whose published size already matches;
- resumes partial files with HTTP range requests;
- checks the final file size;
- optionally verifies checksums with `--verify`; and
- automatically retries over IPv4 after a no-route error.

Check the local dataset status without downloading:

```bash
python3 data/download_dataset.py --check-only
```

### Expected files

| Dataset | Base vectors | Queries | Ground truth |
|---|---|---|---|
| SIFT1M | `data/dataset/sift/base.fvecs` | `data/dataset/sift/query.fvecs` | `data/dataset/sift/gt.ivecs` |
| LAION | `data/dataset/laion1m/100k/laion_base.fvecs` | `data/dataset/laion1m/laion_query.fvecs` | `data/dataset/laion1m/100k/gt.ivecs` |
| TripClick | `data/dataset/trip_distilbert/passages.fvecs` | `data/dataset/trip_distilbert/queries.fvecs` | `data/dataset/trip_distilbert/gt_10.ivecs` |
| MS MARCO | `data/dataset/msmarco_bert/passages.fvecs` | `data/dataset/msmarco_bert/queries.fvecs` | `data/dataset/msmarco_bert/gt_10.ivecs` |
| SIFT100M | `data/dataset/sift100m/base.bin` | `data/dataset/sift100m/query.bin` | `data/dataset/sift100m/gt.bin` |

TripClick and MS MARCO also require the original text and qrels files:

```text
data/dataset/trip_distilbert/benchmark_tsv/documents/docs.tsv
data/dataset/trip_distilbert/benchmark_tsv/topics/topics.head.val.tsv
data/dataset/trip_distilbert/benchmark_tsv/qrels/qrels.dctr.head.val.tsv

data/dataset/msmarco_bert/passages/collection.tsv
data/dataset/msmarco_bert/passages/queries.dev.small.tsv
data/dataset/msmarco_bert/passages/qrels.dev.small.tsv
```

### Dataset properties

The four main experiments follow the dataset sizes, query workloads, metrics,
and HNSW settings used by Compass.

| Dataset | Database vectors | Dimension | Evaluated queries | Metric |
|---|---:|---:|---:|---|
| SIFT1M | 1,000,000 | 128 | 10,000 | Recall@10 |
| LAION | 100,000 | 512 | 1,000 | Recall@10 |
| TripClick | 1,523,871 | 768 | 1,175 | MRR@10 |
| MS MARCO | 8,841,823 | 768 | 6,980 | MRR@10 |
| SIFT100M | 100,000,000 | 128 | 10,000 | Recall@10 |

The code searches upward from the current directory for `data/dataset/`.
An explicit data root can also be selected:

```bash
export MESS_DATA_ROOT=/absolute/path/to/MESS-main/data
```

## IsoHash Initialization

MESS expects one `isohash_weights.bin` file in each dataset directory. Generate
all five files with:

```bash
python3 initialize_isohash.py
```

The paper configuration is:

| Parameter | Value |
|---|---:|
| Total hash length | 8,192 bits |
| Maximum training sample | 100,000 vectors per dataset |
| Random seed | 42 |
| Threshold batch | 64 bits |

The generated files are:

```text
data/dataset/sift/isohash_weights.bin
data/dataset/laion1m/isohash_weights.bin
data/dataset/trip_distilbert/isohash_weights.bin
data/dataset/msmarco_bert/isohash_weights.bin
data/dataset/sift100m/isohash_weights.bin
```

Existing valid files are skipped. Regenerate them with:

```bash
python3 initialize_isohash.py --force
```

Train selected datasets or override the training parameters:

```bash
python3 src/util/LSH/train_isohash.py \
  --datasets sift laion \
  --hash-bits 8192 \
  --max-samples 100000 \
  --seed 42
```

The value passed to `--hash-bits` during training must equal the
`--hash_bits` value used by the server and client.

## Experimental Configuration

The values in this section are the paper configurations. The executable
defaults are convenience values and must not be treated as the authoritative
experimental settings.

### Parameter names

The symbol \(M\) is overloaded in the HNSW literature. MESS uses the following
unambiguous terminology:

| Paper notation | Command-line option | Paper value | Meaning |
|---|---|---:|---|
| \(M_{\mathrm{graph}}\) | `--num_tables` | 64 | Number of graph shards |
| \(t\) | `--graphs_per_vector` | 16 | Number of shards containing each database item |
| \(\kappa\) | `--hash_bits` | 8,192 | Total IsoHash code length before splitting |
| \(\kappa_s\) | derived | 128 | Code length searched in one shard: \(8192/64\) |
| \(M_{\mathrm{hnsw}}\) | `--M` | dataset-specific | Maximum HNSW connectivity parameter |
| `efConstruction` | `--efConstruction` | 300 | HNSW construction breadth |
| \(K\) | `--top_k` | 10 | Final result size |
| \(P\) | `--candidate_pool` | dataset-specific | Aggregate candidate budget across all shards |

In particular:

```text
--num_tables 64
```

means 64 graph shards, whereas:

```text
--M 64
```

means an HNSW connectivity value of 64. These options are not interchangeable.

The implementation requires:

```text
hash_bits % (num_tables * 8) == 0
```

The paper setting `8192 / 64 = 128` therefore gives a byte-aligned 128-bit code
in every shard.

### Fixed MESS and No-Noise settings

Unless a particular experiment varies one of these parameters, all four main
datasets use:

| Parameter | MESS | No-Noise |
|---|---:|---:|
| Graph shards \(M_{\mathrm{graph}}\) | 64 | 64 |
| Shards per item \(t\) | 16 | 16 |
| Total IsoHash length | 8,192 bits | 8,192 bits |
| Per-shard binary-code length | 128 bits | 128 bits |
| `efConstruction` | 300 | 300 |
| Final result size | 10 | 10 |
| Storage flip probability `--p_data` | 0.08 | 0 |
| Permanent query flip probability `--p_base` | 0.08 | 0 |
| Instantaneous query flip probability `--p_inst` | 0.08 | 0 |
| Multi-graph routing | enabled | enabled |
| Encrypted payload transfer | enabled | enabled |
| Client duplicate removal | enabled | enabled |
| Client exact reranking | enabled | enabled |

The implementation does not hard-code the PRR and IRR probabilities. The
values above are the paper's experimental operating point; users can configure
`--p_base` and `--p_inst` according to the desired privacy--utility trade-off.

Perturbation ablations and privacy--quality sweeps override the three flip
probabilities explicitly. No-Noise changes only the randomized perturbation;
it retains the same IsoHash encoding, routing, multi-graph construction,
payload encryption, and reranking pipeline.

### Candidate-pool sweeps

The candidate pool is an aggregate budget across the 64 shards. The client
computes:

```text
per_shard_pool = max(1, floor(candidate_pool / num_tables))
```

The experimental ranges are:

| Dataset | No-Noise aggregate pool | No-Noise per-shard pool | MESS aggregate pool | MESS per-shard pool |
|---|---:|---:|---:|---:|
| SIFT1M | 250--400 | 3--6 | 1,400--1,600 | 21--25 |
| LAION | 200--300 | 3--4 | 700--800 | 10--12 |
| TripClick | 900--1,050 | 14--16 | 2,900--3,200 | 45--50 |
| MS MARCO | 1,400--1,600 | 21--25 | 6,000--6,500 | 93--101 |

The larger MESS pools compensate for rank distortion introduced by
storage-side and query-side perturbation. Every quality--latency point must
report its exact `--candidate_pool` value rather than only the range.

The server and client should be started with the same aggregate candidate
pool. The request sent by the client contains the effective per-shard value.

### Compass-aligned HNSW and workload settings

The dataset-dependent graph and search parameters follow the original Compass
evaluation:

| Dataset | \(M_{\mathrm{hnsw}}\) | Compass `ef` | Compass `efspec` | Compass `efn` | Queries | Metric |
|---|---:|---:|---:|---:|---:|---|
| SIFT1M | 64 | 20 | 4 | 12 | 10,000 | Recall@10 |
| LAION | 64 | 10 | 2 | 12 | 1,000 | Recall@10 |
| TripClick | 128 | 36 | 6 | 24 | 1,175 | MRR@10 |
| MS MARCO | 128 | 48 | 8 | 24 | 6,980 | MRR@10 |

`efspec` and `efn` are Compass-specific traversal parameters and are not MESS
options. MESS uses the same dataset-dependent HNSW connectivity and passes the
corresponding `ef` value through `--efSearch`; its returned candidate count is
controlled separately by the aggregate candidate pool. For MESS and No-Noise,
`efConstruction` is fixed to 300 on every dataset.

All other Compass baseline parameters are taken directly from the original
Compass paper and artifact rather than retuned in this repository. In
particular, the Compass ORAM configuration is:

```text
Z = 32
S = 64
A = 36
```

### Hardware and network model

The main comparison follows the Compass experimental environment:

| Role | Machine |
|---|---|
| Client | Google Cloud `n2-standard-8`: 8 vCPUs, 32 GB memory |
| Server | Google Cloud `n2-highmem-64`: 64 vCPUs, 512 GB memory |

| Network | Bandwidth | Round-trip latency |
|---|---:|---:|
| Fast / same region | 3 Gbps | 1 ms |
| Slow / cross region | 400 Mbps | 80 ms |

Use the same machines, thread counts, network shaping, query set, and metric
for every compared system. Network shaping should be applied consistently to
both directions of the client--server connection.

### Baseline policy

- **Plaintext-HNSW:** uses the original float embeddings and the
  dataset-specific Compass HNSW settings. It is the non-private quality and
  efficiency reference.
- **No-Noise:** uses the complete MESS pipeline with all three flip
  probabilities set to zero and the smaller candidate-pool ranges listed
  above.
- **Compass:** uses the configuration reported by its paper and public
  artifact, including its dataset-specific HNSW, traversal, PQ, ORAM, hardware,
  and network settings.
- **HE-Cluster:** uses the results and configurations reported in the Compass
  evaluation. Do not synthesize or extrapolate an unreported dataset result.

## Running MESS

The server builds the graph shards before accepting requests. Start it first,
then start the client in a second terminal. Always pass an explicit common
port because the built-in server and client defaults differ.

### SIFT1M

This example uses a candidate pool of 1,500, which lies within the paper range.

Server:

```bash
./build/server_origin \
  --dataset sift \
  --M 64 \
  --efConstruction 300 \
  --efSearch 20 \
  --top_k 10 \
  --candidate_pool 1500 \
  --hash_bits 8192 \
  --num_tables 64 \
  --graphs_per_vector 16 \
  --p_data 0.08 \
  --bind_ip 0.0.0.0 \
  --port 9090
```

Client:

```bash
./build/client_origin \
  --dataset sift \
  --metric recall \
  --test_queries 10000 \
  --top_k 10 \
  --candidate_pool 1500 \
  --hash_bits 8192 \
  --num_tables 64 \
  --p_base 0.08 \
  --p_inst 0.08 \
  --server_ip 127.0.0.1 \
  --port 9090
```

### LAION

This example uses a candidate pool of 700.

Server:

```bash
./build/server_origin \
  --dataset laion \
  --M 64 \
  --efConstruction 300 \
  --efSearch 10 \
  --top_k 10 \
  --candidate_pool 700 \
  --hash_bits 8192 \
  --num_tables 64 \
  --graphs_per_vector 16 \
  --p_data 0.08 \
  --bind_ip 0.0.0.0 \
  --port 9090
```

Client:

```bash
./build/client_origin \
  --dataset laion \
  --metric recall \
  --test_queries 1000 \
  --top_k 10 \
  --candidate_pool 700 \
  --hash_bits 8192 \
  --num_tables 64 \
  --p_base 0.08 \
  --p_inst 0.08 \
  --server_ip 127.0.0.1 \
  --port 9090
```

### TripClick

This example uses a candidate pool of 3,000.

Server:

```bash
./build/server_origin \
  --dataset trip \
  --M 128 \
  --efConstruction 300 \
  --efSearch 36 \
  --top_k 10 \
  --candidate_pool 3000 \
  --hash_bits 8192 \
  --num_tables 64 \
  --graphs_per_vector 16 \
  --p_data 0.08 \
  --bind_ip 0.0.0.0 \
  --port 9090
```

Client:

```bash
./build/client_origin \
  --dataset trip \
  --metric mrr \
  --test_queries 1175 \
  --top_k 10 \
  --candidate_pool 3000 \
  --hash_bits 8192 \
  --num_tables 64 \
  --p_base 0.08 \
  --p_inst 0.08 \
  --server_ip 127.0.0.1 \
  --port 9090
```

### MS MARCO

This example uses a candidate pool of 6,500.

Server:

```bash
./build/server_origin \
  --dataset msmarco \
  --M 128 \
  --efConstruction 300 \
  --efSearch 48 \
  --top_k 10 \
  --candidate_pool 6500 \
  --hash_bits 8192 \
  --num_tables 64 \
  --graphs_per_vector 16 \
  --p_data 0.08 \
  --bind_ip 0.0.0.0 \
  --port 9090
```

Client:

```bash
./build/client_origin \
  --dataset msmarco \
  --metric mrr \
  --test_queries 6980 \
  --top_k 10 \
  --candidate_pool 6500 \
  --hash_bits 8192 \
  --num_tables 64 \
  --p_base 0.08 \
  --p_inst 0.08 \
  --server_ip 127.0.0.1 \
  --port 9090
```

### No-Noise ablation

Use the same command as MESS, but set:

```text
server: --p_data 0
client: --p_base 0 --p_inst 0
```

and select the appropriate No-Noise pool from the candidate-pool table. Do not
change the hash length, graph count, routing multiplicity, HNSW construction,
encryption, or reranking configuration.

For example, SIFT1M with a pool of 300 uses:

```bash
# Server-side differences
--candidate_pool 300 --p_data 0

# Client-side differences
--candidate_pool 300 --p_base 0 --p_inst 0
```

### SIFT100M

SIFT100M uses `base.bin`, `query.bin`, and `gt.bin` in DiskANN binary format.
The fixed multi-graph parameters remain:

```text
hash_bits=8192
num_tables=64
graphs_per_vector=16
per_shard_hash_bits=128
efConstruction=300
top_k=10
```

Set the candidate pool and query perturbation probability to the operating
point being reproduced. For example:

```bash
./build/server_origin \
  --dataset sift100m \
  --M 64 \
  --efConstruction 300 \
  --efSearch 64 \
  --top_k 10 \
  --candidate_pool 8000 \
  --hash_bits 8192 \
  --num_tables 64 \
  --graphs_per_vector 16 \
  --p_data 0.08 \
  --bind_ip 0.0.0.0 \
  --port 9090
```

```bash
./build/client_origin \
  --dataset sift100m \
  --metric recall \
  --test_queries 10000 \
  --top_k 10 \
  --candidate_pool 8000 \
  --hash_bits 8192 \
  --num_tables 64 \
  --p_base 0.08 \
  --p_inst 0.08 \
  --server_ip 127.0.0.1 \
  --port 9090
```

SIFT100M candidate-pool and perturbation sweeps must report the exact values
used for each point.

## Plaintext HNSW Baseline

Start the plaintext server:

```bash
./build/hnswlib_server \
  --dataset sift \
  --M 64 \
  --efConstruction 300 \
  --ef 20 \
  --top_k 10 \
  --threads 16 \
  --bind_ip 0.0.0.0 \
  --port 9091
```

Run the client:

```bash
./build/hnswlib_client \
  --dataset sift \
  --metric recall \
  --k 10 \
  --ef 20 \
  --test_queries 10000 \
  --threads 16 \
  --server_ip 127.0.0.1 \
  --port 9091
```

Use the dataset-specific `M`, `ef`, query count, and metric from the
Compass-aligned table for the other datasets.

## Dynamic Insertion and Deletion

Start `server_origin` with the target dataset and normal MESS parameters.
Insert a generated vector:

```bash
./build/update_example \
  --ip 127.0.0.1 \
  --port 9090 \
  --op insert \
  --dim 128 \
  --label -1 \
  --seed 1
```

Use the correct dataset dimension: 128 for SIFT, 512 for LAION, and 768 for
TripClick or MS MARCO. A negative label asks the server to allocate a new
label.

Delete the returned label:

```bash
./build/update_example \
  --ip 127.0.0.1 \
  --port 9090 \
  --op delete \
  --label 1000001
```

Deletion marks the corresponding graph nodes inactive. The server excludes
inactive labels from returned candidate sets.

## Security and Linkage Experiments

The attack package evaluates:

- fixed-IsoHash XDP;
- exact randomized-response privacy-loss distributions;
- PRR/IRR access- and search-pattern accounting;
- MDS-based cross-shard linkage;
- topology-based linkage;
- graph-signal graph matching (GSGM);
- joint cycle-consistent linkage; and
- linkage-aware reconstructed-vector XDP.

Install and validate:

```bash
cd src/attack
python3 -m pip install -r requirements.txt
python3 run.py --dataset sift --check
```

Run the standard SIFT analysis:

```bash
python3 run.py --dataset sift --profile standard --mode all
```

Run all available datasets:

```bash
bash scripts/run_all_standard.sh
```

See [`src/attack/README.md`](src/attack/README.md) for the complete experiment
definitions, profiles, parameters, statistical guarantees, and output files.

## Command-Line Reference

### `server_origin`

| Option | Meaning |
|---|---|
| `--dataset` | `sift`, `laion`, `trip`, `msmarco`, or `sift100m` |
| `--M` | HNSW connectivity \(M_{\mathrm{hnsw}}\), not the number of shards |
| `--efConstruction` | HNSW index-construction breadth |
| `--efSearch` | HNSW search breadth |
| `--top_k` | Final result count |
| `--candidate_pool` | Aggregate candidate budget |
| `--hash_bits` | Total IsoHash code length |
| `--num_tables` | Number of graph shards |
| `--graphs_per_vector` | Number of shards selected for each database item |
| `--p_data` | Storage-side bit-flip probability |
| `--build_batch_size` | Number of vectors encoded per construction batch |
| `--omp_threads` | OpenMP thread count; `0` uses the runtime default |
| `--bind_ip` | Listening IPv4 address |
| `--port` | Listening TCP port |
| `--base_path` | Override the default base-vector file |

### `client_origin`

| Option | Meaning |
|---|---|
| `--dataset` | `sift`, `laion`, `trip`, `msmarco`, or `sift100m` |
| `--metric` | `recall` or `mrr` |
| `--test_queries` | Number of queries to evaluate |
| `--top_k` | Final result count |
| `--candidate_pool` | Aggregate candidate budget; must match the experiment |
| `--hash_bits` | Total IsoHash code length |
| `--num_tables` | Number of query shard reports |
| `--p_base` | Permanent randomized-response flip probability |
| `--p_inst` | Instantaneous randomized-response flip probability |
| `--use_query_batch` | Enable or disable batched query messages |
| `--query_batch_size` | Queries per batch |
| `--server_ip` | Server IPv4 address |
| `--port` | Server TCP port |
| `--omp_threads` | OpenMP thread count |
| `--query_path` | Override the default query-vector file |
| `--gt_path` | Override the default ground-truth file |
| `--collection_tsv` | Override the document collection |
| `--queries_tsv` | Override the query text file |
| `--qrels_tsv` | Override the relevance judgments |

## Reproducibility Checklist

Before recording a result, verify:

1. all required dataset files pass `--check-only`;
2. every `isohash_weights.bin` was trained with 8,192 bits;
3. server and client both use `--hash_bits 8192`;
4. server and client both use `--num_tables 64`;
5. the server uses `--graphs_per_vector 16`;
6. `--M` is the correct dataset-specific HNSW connectivity;
7. `--efConstruction 300` is set explicitly;
8. the server and client use the same candidate pool and TCP port;
9. MESS uses the declared storage, PRR, and IRR probabilities;
10. No-Noise sets all three probabilities to zero;
11. the correct query count and Recall@10 or MRR@10 metric are used; and
12. hardware and network shaping match the Compass comparison.

## Important Notes

- MESS routes database items, not queries, to \(t=16\) selected shards. A
  query report is generated for each of the 64 searchable shards.
- A larger candidate pool increases encrypted response size and client
  reranking work.
- The paper parameters should always be supplied explicitly; executable
  defaults may change during development.
- The source contains research-prototype key material for local evaluation.
  Replace it with proper key provisioning before any security-sensitive use.
- Do not present the attack experiments as replacements for the formal XDP
  guarantees. They quantify the leakage achieved by the evaluated inference
  procedures.

## References

- J. Zhu, L. Patel, M. Zaharia, and R. A. Popa,
  [Compass: Encrypted Semantic Search with High Accuracy](https://www.usenix.org/system/files/osdi25-zhu-jinhao.pdf),
  OSDI 2025.
- Compass artifact: <https://github.com/Clive2312/compass>
- SIFT100M-DiskANN: <https://huggingface.co/datasets/Nanvivi/SIFT100M-DiskANN>
