# MMM-Trend

MMM-Trend is a multilingual, multimodal misinformation dataset for tempo-spatial analysis. It contains fact-checked claims and news items in eight Indian languages and English, each with enrichment (English translation, summaries, keywords), a topic, a location and candidate images.

This repository holds the dataset, the code used to build it, the benchmark code and the benchmark results reported in the paper.

## Dataset at a glance

| | |
|---|---|
| Instances | 30,115 |
| Real / Fake | 18,687 / 11,428 |
| Languages | English, Hindi, Bengali, Gujarati, Telugu, Tamil, Assamese, Odia, Malayalam |
| Time span | 2012–2024 |
| Topics | 7 |
| Candidate images per instance | up to 10 |

## Repository layout

```
data/
  complete_dataset_v1.2.xlsx    the dataset (sheet "Complete-Dataset", one row per instance)
  articles_text/                article body for the 22,849 instances that have one, named by the File column
code/
  pipeline/                     enrichment, image retrieval, BLIP/CLIP matching, annotation survey
  topics/                       fusion comparison, choice of K, topic clustering
  benchmark/                    the nine baselines, temporal and spatial protocols
results/
  temporal_2024/  temporal_2023/  spatial/    reported numbers, one file per model plus summary.csv
```

## Dataset fields

| Field | Description |
|---|---|
| `dates`, `File` | Date (dd/mm/yyyy) and identifier of the instance |
| `Title`, `Translated_title` | Claim title in the base language and in English |
| `Language` | Base language of the instance |
| `Authenticity` | Veracity label: `Real` or `Fake` |
| `Summary`, `Summary(EN)` | Abstractive summary in the base language and in English |
| `Keywords (EN)`, `Keywords (Base Language)` | KeyBERT keywords |
| `Topics` | Assigned latent topic (one of seven) |
| `Location`, `Location_std` | Raw location and standardised region |
| `Precise Locations` | Place names extracted from the text |
| `Focus`, `Target`, `Stance Detection`, `Political Influence`, `Authority Involvement`, `Scientific Data Mentioned`, `Statistical Data Mentioned`, `Source Mentioned` | Claim attributes, available for the 2,298 Dhoroni instances only |
| `Image 1 ID` … `Image 10 ID` | Candidate images retrieved for the instance |
| `Best Match Image (BLIP)`, `Best Match Image (CLIP)` | Top-ranked candidate by each model |
| `BLIP-CLIP Matching` | Whether BLIP and CLIP pick the same image |

**Labels.** Source labels were consolidated into two classes: True/Real → `Real`; False/Fake → `Fake`; instances marked "Maybe Real" or "Maybe Fake" were assigned to the minority class of their year-quarter.

**Article text.** 22,849 instances have a full article body in `data/articles_text/`. For the remaining instances the title and summary in the spreadsheet are the available text.

**Regions.** The spatial benchmark groups `Location_std` into four regions; `Pan_India` and `International` are never held out.

| Region | `Location_std` values | Instances |
|---|---|---|
| R1 South | Tamil_Nadu, Kerala, Telangana_Andhra, Karnataka | 8,944 |
| R2 East | West_Bengal, Odisha, Bihar, Assam, Northeast, Jharkhand, Bangladesh | 5,728 |
| R3 West | Gujarat, Maharashtra, Rajasthan, Goa | 3,580 |
| R4 North/Central | Delhi, Uttar_Pradesh, Jammu_Kashmir, Punjab, Madhya_Pradesh, Uttarakhand, Haryana, Chhattisgarh, Himachal_Pradesh | 2,838 |
| Always in training | Pan_India, International | 9,025 |

## Images

Image files are not stored in this repository because of their size. They are available for anonymous review at:

https://drive.google.com/drive/folders/1Oon-nUdd_ORsOmrN4-D1ErhjMNmmKy69?usp=drive_link

File names match the values in the `Image 1 ID` … `Image 10 ID` and `Best Match Image` columns. To run the image and text+image baselines or the topic code, place the image files in a folder named `images/` at the repository root.

## Benchmark

Nine baselines are evaluated: BERT, EANN and FTT on text; CLIP, BLIP and Qwen2-VL-2B on images; CLIP, BLIP and Qwen2-VL-2B on text+image.

- **Temporal (2024):** train on instances dated up to 2023-Q4, test on each 2024 quarter.
- **Temporal (2023):** train on instances dated before 2023, test on each 2023 quarter.
- **Spatial:** leave-one-region-out over R1–R4.

In every protocol 10% of the training instances are held out for validation (seed 2378) and used for early stopping on Macro-F1. Decision thresholds are fixed at 0.40 for text models and 0.30 for image and text+image models; they are not tuned on test data.

Text models fine-tune the last layer of `bert-base-multilingual-cased`. Image and text+image models train an MLP on frozen features from `openai/clip-vit-base-patch32`, `Salesforce/blip-image-captioning-base` and `Qwen/Qwen2-VL-2B`.

Run from `code/benchmark/`. Outputs are written to `bench_results/` at the repository root.

| Result | Command |
|---|---|
| BERT, 2024 | `python bench_bert_040.py` |
| BERT, 2023 | `python bench_bert_year.py 2023` |
| EANN, 2024 / 2023 | `python bench_eann.py` / `python bench_eann.py 2023` |
| FTT, 2024 / 2023 | `python ftt_prepare.py`, then the reweighting step below, then `python bench_ftt.py` (add `2023` to both for the 2023 run) |
| Image and text+image, 2024 | `python bench_mm.py <model>` |
| Image and text+image, 2023 | `python bench_mm_year.py <model> 2023` (after the 2024 run, which caches the features) |
| Spatial | `python bench_region.py <model>` |

`<model>` is one of `clip_image`, `clip_vl`, `blip_image`, `blip_vl`, `qwen_image`, `qwen_vl`; `bench_region.py` also accepts `bert` and `eann`. FTT is a temporal forecasting method and is not run on the spatial splits.

**FTT reweighting.** `ftt_prepare.py` writes the splits in the layout expected by the original FTT implementation (Hu et al., 2023). Run that implementation's forecasting step on them to obtain the instance weights under `ftt_reweight/roll_seasonal_user_data/`; `bench_ftt.py` then trains with those weights.

## Topics

Run from `code/topics/` (requires `images/`).

- `topic_fusion_5seed.py` compares four ways of combining text and image embeddings by topic coherence (NPMI) over five seeds.
- `inertia_elbow.py` computes KMeans inertia and the silhouette coefficient for K = 2 to 20.
- `cluster_topics.py` clusters the mean-fused CLIP embeddings of all instances into K = 7 groups.

## Pipeline

`code/pipeline/` contains the scripts used to build the dataset: summarisation and keyword extraction, place extraction, image retrieval through SerpAPI (supply your own API key), BLIP/CLIP image–text matching and generation of the annotation survey sheets. Input and output file names at the top of each script should be set to your own paths.

## Annotations

'data/annotations/' contains the annotation files used for assessing agreement among annotators on the candidate image–text matches. The files retain the instance identifiers and candidate image IDs, together with annotator-assigned scores. The merged annotation file consolidates the individual annotations and Kappa calculation file includes the calculated inter-annotator agreement statistics, i.e. Fleiss’ kappa.

These files support inspection of the annotation data and reproducibility of the agreement analysis. See the files in 'data/annotations/' for the individual annotator annotations, merged annotations and agreement calculations.

### Inter-Annotator Agreement

Inter-annotator agreement was calculated using Fleiss' kappa across the candidate image–text items rated by three annotators on a five-category scale.

| Calculation step | Formula / parameter | Value / explanation |
|---|---|---|
| Number of items (\(N\)) | `45937` | Number of image–text items with all three ratings |
| Number of annotators (\(m\)) | `3` | Number of annotators |
| Categories | `1, 2, 3, 4, 5` | Categorical rating labels |
| Per-item agreement | \(P_i = \frac{\sum_{j=1}^{5} n_{ij}^{2}-m}{m(m-1)}\) | \(n_{ij}\) is the number of annotators assigning category \(j\) to item \(i\) |
| Observed agreement | \(\bar{P} = \frac{1}{N}\sum_{i=1}^{N} P_i\) | `0.802903977` |
| Category proportion | \(p_j = \frac{n_j}{N \times m}\) | \(n_j\) is the total number of ratings assigned to category \(j\) |
| Chance agreement | \(P_e = \sum_{j=1}^{5} p_j^2\) | `0.211557991` |
| **Fleiss' kappa** | \(\kappa = \frac{\bar{P}-P_e}{1-P_e}\) | **`0.750018365`** |

Here, \(N\) denotes the number of items, \(m\) the number of annotators, and \(n_j\) the total number of ratings in category \(j\). Fleiss' kappa measures agreement among multiple annotators while accounting for the agreement expected by chance.

**Reproducibility:** The annotation files and the merged annotation data are available in `data/annotations/`, together with the calculation details.

**Note:** Report this coefficient as empirical inter-annotator agreement only after verifying that the underlying ratings are genuine annotator judgements. 

## Results

`results/` contains the numbers reported in the paper's three benchmark tables: one JSON file per model and a `summary.csv` per table, with Macro-F1, accuracy and class-wise F1 for each quarter or region and their average.

## Requirements

Python 3.10 or later. See `requirements.txt`.

## Ethical note

The dataset contains misinformation collected for academic research. It should be used for research on detecting and understanding misinformation and handled responsibly.

## Anonymity

This repository is prepared for double-blind review. Identifying information has been removed.
