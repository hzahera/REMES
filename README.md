# REMES: Relatedness-based Multi-Entity Summarization

[![Paper](https://img.shields.io/badge/Paper-IJCAI%202017-blue)](https://www.ijcai.org/proceedings/2017/0147.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-green.svg)](https://www.python.org/)

Official implementation of the paper:

> **Relatedness-based Multi-Entity Summarization**  
> Kalpa Gunaratna, Amir Hossein Yazdavar, Krishnaprasad Thirunarayan, Amit Sheth, Gong Cheng  
> *Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI 2017)*  
> [https://www.ijcai.org/proceedings/2017/0147.pdf](https://www.ijcai.org/proceedings/2017/0147.pdf)

---

## Overview

REMES is an approach for generating entity summaries over a **collection of entities** from a knowledge graph (e.g., DBpedia). Unlike prior standalone entity summarization systems (RELIN, FACES, LinkSum), REMES considers **multiple entities simultaneously** and produces summaries that maximize:

1. **Inter-entity relatedness** — facts selected across entities should be related to each other.
2. **Intra-entity importance** — selected facts should be important within each individual entity.
3. **Intra-entity diversity** — selected facts should be diverse within each entity to avoid redundancy.

The problem is formulated as a **Quadratic Multidimensional Knapsack Problem (QMKP)** and solved efficiently using a memory-based adaptation of the **GRASP (Greedy Randomized Adaptive Search Procedures)** algorithm.

### Key Features

- Graph-based and semantics-based relatedness measures using **RDF2Vec** and **WordNet**
- Adapts QMKP with a **Restricted Candidate List (RCL)** threshold for diversity
- Evaluated on two benchmark datasets: **Wikinews** and **AQUAINT**
- Outperforms FACES and RELIN on qualitative and quantitative evaluation metrics

---

## Repository Structure

```
REMES/
├── data/                    # Benchmark datasets
│   ├── wikinews/            # Wikinews dataset (20 documents)
│   └── aquaint/             # AQUAINT dataset (10 documents)
├── models/                  # Pre-trained RDF2Vec embeddings
│   └── rdf2vec_dbpedia/     # RDF2Vec model trained on DBpedia 2016-04
├── src/                     # Source code
│   ├── remes.py             # Main REMES summarization pipeline
│   ├── grasp.py             # Memory-based GRASP optimization
│   ├── relatedness.py       # Semantic & graph-based relatedness measures
│   ├── importance.py        # Feature importance scoring (tf-idf based rank)
│   ├── diversity.py         # Intra-entity diversity computation
│   └── utils.py             # Helper utilities (KG access, preprocessing)
├── baselines/               # Baseline implementations for comparison
│   ├── relin.py             # RELIN baseline
│   └── faces.py             # FACES baseline
├── evaluation/              # Evaluation scripts
│   ├── qualitative.py       # Likert-scale questionnaire analysis
│   └── quantitative.py      # UCI / UMass coherence evaluation
├── requirements.txt         # Python dependencies
├── config.yaml              # Configuration file (hyperparameters)
├── run_remes.py             # Entry point to run REMES
└── README.md
```

---

## Approach

REMES operates in the following stages:

```
Input: Collection of entities E = {e1, e2, ..., en} from a KG
           |
           v
1. Feature Extraction
   - Retrieve feature sets FS(e) for each entity from DBpedia
   - Compute importance: rank(f) = Inf(f) * Po(Val(f))
           |
           v
2. Relatedness Computation
   - Semantic relatedness: SemRel via WordNet (Jaccard on hypernyms)
   - Graph relatedness:    GraphRel via RDF2Vec cosine similarity
   - Combined:             r(fi, fj) = (SemRel + GraphRel) / 2
           |
           v
3. Optimization via GRASP (adapted QMKP)
   - Profit matrix with alpha (importance), beta (intra-entity diversity),
     gamma (inter-entity relatedness)
   - Restricted Candidate List (RCL) with threshold eta for diversity
           |
           v
Output: Entity summaries Summ(e1), Summ(e2), ..., Summ(en)
        with maximized inter-entity relatedness
```

---

## Installation

### Prerequisites

- Python 3.7+
- Java 8+ (for DBpedia SPARQL endpoint, optional)
- ~4 GB disk space for RDF2Vec embeddings

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/hzahera/REMES.git
cd REMES

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the RDF2Vec model pre-trained on DBpedia 2016-04
#    (Place the model files under models/rdf2vec_dbpedia/)
#    Model available at: http://data.dws.informatik.uni-mannheim.de/rdf2vec/
```

### Dependencies (`requirements.txt`)

```
numpy>=1.18.0
scipy>=1.4.0
gensim>=3.8.0          # For RDF2Vec word2vec model loading
nltk>=3.5              # For WordNet-based semantic similarity
SPARQLWrapper>=1.8.5   # For DBpedia SPARQL queries
pyyaml>=5.3            # For config.yaml parsing
tqdm>=4.45.0           # Progress bars
```

---

## Configuration

All hyperparameters are set in `config.yaml`:

```yaml
# GRASP algorithm parameters
grasp:
  tau: 1           # Greedy ranking component weight
  phi: 0.5         # Candidate feature selection weight
  sigma: 5         # Number of GRASP iterations
  lambda: 5        # Local search improvement iterations

# Profit matrix weights
profit:
  alpha: 2         # Weight for feature importance (intra-entity)
  beta: 1          # Weight for intra-entity diversity penalty
  gamma: 0.45      # Weight for inter-entity relatedness

# Summary length
summary:
  knapsack_size: 5  # Number of facts per entity summary (k)

# Restricted Candidate List threshold
eta: 0.45

# Knowledge graph
kg:
  endpoint: "https://dbpedia.org/sparql"
  dataset_version: "2016-04"

# Paths
paths:
  rdf2vec_model: "models/rdf2vec_dbpedia/"
  wordnet_db: "models/wordnet/"
  data_dir: "data/"
```

---

## How to Run

### Quick Start

```bash
# Run REMES on the Wikinews dataset
python run_remes.py --dataset wikinews --output results/wikinews_summaries.json

# Run REMES on the AQUAINT dataset
python run_remes.py --dataset aquaint --output results/aquaint_summaries.json
```

### Run with Custom Entity List

```bash
python run_remes.py \
  --entities "dbr:Apple_Computer" "dbr:Steve_Jobs" \
  --summary_size 5 \
  --output results/custom_summaries.json
```

### Full Options

```
usage: run_remes.py [-h] [--dataset {wikinews,aquaint}]
                   [--entities ENTITIES [ENTITIES ...]]
                   [--summary_size SUMMARY_SIZE]
                   [--config CONFIG]
                   [--output OUTPUT]
                   [--alpha ALPHA] [--beta BETA] [--gamma GAMMA]
                   [--eta ETA] [--sigma SIGMA]

arguments:
  --dataset          Benchmark dataset to use: wikinews or aquaint
  --entities         Space-separated list of DBpedia entity URIs
  --summary_size     Number of facts per entity summary (default: 5)
  --config           Path to config YAML file (default: config.yaml)
  --output           Output file path for generated summaries (JSON)
  --alpha            Importance weight in profit matrix (default: 2)
  --beta             Intra-entity diversity weight (default: 1)
  --gamma            Inter-entity relatedness weight (default: 0.45)
  --eta              RCL threshold for diversity (default: 0.45)
  --sigma            Number of GRASP iterations (default: 5)
```

### Programmatic Usage

```python
from src.remes import REMES

# Initialize REMES with config
remes = REMES(config_path="config.yaml")

# Define entity collection
entities = [
    "http://dbpedia.org/resource/Apple_Computer",
    "http://dbpedia.org/resource/Steve_Jobs",
]

# Generate multi-entity summaries
summaries = remes.summarize(entities, summary_size=5)

# Print results
for entity, facts in summaries.items():
    print(f"\nSummary for {entity}:")
    for fact in facts:
        print(f"  {fact['property']}: {fact['value']}")
```

---

## Datasets

REMES was evaluated on two benchmark datasets from the entity linking literature:

| Dataset   | Documents | Entities per doc | Source |
|-----------|-----------|-----------------|--------|
| Wikinews  | 20        | varies          | [Wikinews corpus](https://en.wikinews.org/) |
| AQUAINT   | 10        | varies          | AQUAINT newswire corpus |

The datasets contain news documents along with their linked DBpedia entities. Entity summaries are generated for all entities in each document.

---

## Evaluation

### Qualitative Evaluation (Likert Scale)

Human judges answered 5 questions about the summaries on a 1–5 Likert scale:

| Q  | Question |
|----|----------|
| Q1 | Summaries assist in understanding relationships between entities |
| Q2 | Facts in each summary are diverse |
| Q3 | Summaries helped better understand the document |
| Q4 | Summaries provide an overview of the entity collection |
| Q5 | I like the summaries generated |

```bash
# Run qualitative evaluation
python evaluation/qualitative.py --results results/wikinews_summaries.json
```

### Quantitative Evaluation (Semantic Coherence)

Coherence between selected entity summaries measured via:
- **UCI** (PMI-based, Wikipedia sliding window)
- **UMass** (document co-occurrence)

```bash
# Compute UCI and UMass coherence scores
python evaluation/quantitative.py \
  --results results/wikinews_summaries.json \
  --metric uci umass
```

### Reproducing Paper Results

```bash
# Run full experiment pipeline (both datasets, all baselines)
bash scripts/run_all_experiments.sh

# Results will be saved to results/
# Tables will be printed to stdout
```

---

## Results

### Qualitative Results (Wikinews — Mean Likert scores)

| Question | REMES | FACES | RELIN |
|----------|-------|-------|-------|
| Q1       | 3.98  | 3.66  | 2.78  |
| Q2       | 4.12  | 3.93  | 3.79  |
| Q3       | 3.69  | 3.38  | 2.84  |
| Q4       | 3.78  | 3.48  | 2.51  |
| Q5       | 4.08  | 3.72  | 3.18  |

### Quantitative Results (Average Coherence)

| System | Wikinews (UCI) | AQUAINT (UCI) | Wikinews (UMass) | AQUAINT (UMass) |
|--------|---------------|--------------|-----------------|----------------|
| REMES  | **0.064**     | **-0.056**   | **-0.301**      | **-0.257**     |
| FACES  | -0.083        | -0.259       | -0.971          | -0.428         |
| RELIN  | -0.221        | -0.148       | -0.984          | -0.589         |

*Higher UCI/UMass scores indicate more semantically coherent summaries.*

---

## Citation

If you use REMES in your research, please cite:

```bibtex
@inproceedings{gunaratna2017remes,
  title     = {Relatedness-based Multi-Entity Summarization},
  author    = {Gunaratna, Kalpa and Yazdavar, Amir Hossein and Thirunarayan, Krishnaprasad and Sheth, Amit and Cheng, Gong},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17)},
  pages     = {1060--1066},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/147},
  url       = {https://www.ijcai.org/proceedings/2017/0147.pdf}
}
```

---

## Related Work

- **RELIN** [Cheng et al., 2011] — PageRank-based standalone entity summarization
- **FACES** [Gunaratna et al., 2015] — Hierarchical clustering-based entity summarization  
- **LinkSum** [Thalhammer et al., 2016] — Link analysis for entity summarization
- **RDF2Vec** [Ristoski and Paulheim, 2016] — Graph-based entity embeddings used for relatedness

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the paper or code, please open a GitHub issue or contact the original authors:
- Kalpa Gunaratna — Kno.e.sis, Wright State University
- Amir Hossein Yazdavar — Kno.e.sis, Wright State University  
- Paper: [https://www.ijcai.org/proceedings/2017/0147.pdf](https://www.ijcai.org/proceedings/2017/0147.pdf)
