# Tutorial 4 — Multimodal Similarity Analysis

Encode gestures and their co-occurring speech, then ask **what makes two gestures similar**: the object they
depict, who produced them, or the conversation they came from.

**Notebook:** `Extracting_and_Analysing_Multimodal_Representations_Zoom.ipynb`

**Data:** 45 Dutch dyads playing a director–matcher referential game (16 Fribbles × 6 rounds) **over Zoom**,
with mutual visibility manipulated between dyads: `Sym` (both cameras on), `Asym` (one camera on), `AO`
(audio only). 15,324 manually segmented gesture strokes, of which ~10.4k iconic ones carry a referent label
pointing at the object part they depict (a stroke that depicts two parts counts once per part).

## What the notebook does

1. Reads the gesture annotations (ELAN), the body keypoints (MMPose, 50 fps) and the WhisperX transcripts.
2. Encodes the speech that co-occurs with each gesture using **Dutch BERT** (768-d).
3. Encodes each gesture stroke using the pretrained **Multimodal-X** skeleton encoder (128-d).
4. Computes cosine similarity for every pair of gestures and labels each pair: same/different **referent**,
   same/different **speaker**, same/different **dialogue**.
5. Tests the pair types against each other (box, violin, bar and KDE plots with significance tests).
6. Compares the **visibility conditions** — how much cross-speaker similarity survives when partners cannot see
   each other?

## What you need

```
Tutorial_4_Multimodal_Similarity_Analysis/
├── Extracting_and_Analysing_Multimodal_Representations_Zoom.ipynb   ← run this
├── model/ · dialog_utils/ · utils/            helper modules imported by the notebook
├── data/technology_mediated_interaction/
│   ├── 01_ELAN/                               45 × .eaf gesture annotations
│   │   └── tm_interaction_aligned.csv         WhisperX word-level transcripts
│   ├── 02_mmpose/                             90 × .npy keypoint files (one per participant)
│   ├── participant_info_to_share.csv          pair, speaker, visibility condition
│   └── processed_gestures_..._all_embeddings.pkl   pre-computed embeddings (cache)
└── pretrained_models/multimodal-x/            gesture encoder (downloads automatically)
```

The gesture-encoder checkpoint is fetched from
[Google Drive](https://drive.google.com/file/d/14lBbWLK53Ct6nOHV84yy2kXTeDh5I0TS/view?usp=sharing) on first run.

## Setup

The workshop environment ([main README](../README.md)) covers everything needed to run the notebook **from the
cache**. To rebuild the embeddings from the raw annotations you also need:

```bash
pip install pympi-ling spacy
python -m spacy download nl_core_news_sm
```

## Run it

1. Open the notebook in VS Code and select the workshop environment as the kernel.
2. Run the cells from top to bottom.
   - With the cached `.pkl` present (`USE_CACHE = True`), extraction is skipped and the notebook goes straight to
     the analysis — a few minutes.
   - Without it, the notebook re-reads the ELAN files and runs BERT and the gesture encoder over ~10k gestures —
     tens of minutes on a CPU — and writes the cache at the end.
3. Play with the two switches at the bottom of the notebook and re-run:
   - `MODALITY` — `'gesture'`, `'semantic'` (speech) or `'multimodal'` (both concatenated)
   - `PLOT_CONDITION` — `'Sym'`, `'AO'`, `'Asym'` or `None`

**On a laptop:** the full data gives ~9.9M gesture pairs and needs several GB of RAM. If it is slow, set
`MAX_GESTURES = 3000` and `MAX_PAIRS_PER_TYPE = 50_000` in the configuration cell.

## What you should see

With `MODALITY = 'gesture'`, gestures are most similar when they depict the same object *and* come from the same
speaker, and least similar when nothing is shared:

| Comparison | Result |
|---|---|
| same referent vs. different referent (same speaker) | same referent is clearly more similar |
| same referent, same speaker vs. different speaker | a speaker resembles themselves most |
| same referent, same dialogue vs. different dialogue | interlocutors resemble each other more than strangers do |

And the condition comparison (§9), the part this corpus adds:

| Pair type | Sym | AO | Sym vs AO |
|---|---|---|---|
| same referent, **different speaker** (same dialogue) | 0.21 | 0.15 | **** |
| same referent, same speaker *(control)* | 0.47 | 0.48 | n.s. |
| different referent, different speaker *(control)* | 0.12 | 0.12 | n.s. |

Cross-speaker similarity drops when the partners cannot see each other, while a speaker's own consistency does
not — visibility affects alignment *between* people, not gesturing as such.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: pympi` / `spacy` | Only needed when rebuilding — install as above, or keep `USE_CACHE = True` |
| Checkpoint download fails | Download it manually from the Drive link into `pretrained_models/multimodal-x/` |
| `statannotations` missing | Nothing to do — the notebook falls back to its own Welch-test brackets |
| Out of memory / very slow | Lower `MAX_GESTURES` and `MAX_PAIRS_PER_TYPE` |
| File-not-found on the data | Check `DATA_ROOT` in the configuration cell; run the notebook from this folder |

## References

- Ghaleb et al. (2024). *Learning Co-Speech Gesture Representations in Dialogue through Contrastive Learning: An
  Intrinsic Evaluation.* ICMI '24. <https://doi.org/10.1145/3678957.3685707> — the pair types and analyses.
- Ghaleb et al. (2025). *I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal
  Dialogue.* — the Multimodal-X encoder.
- Akamine, Meyer & Özyürek (manuscript). *Lexical alignment and speaker visibility influence gestural alignment
  in conversation.* — the video-mediated corpus.
