# SWiFT-MLP — Semi-Automated Workflows for Facilitating Multimodal Language Processing

**MMSYM 2026 workshop · Leuven, Belgium · 8 September 2026**
Organised by [Esam Ghaleb](https://esamghaleb.github.io/) & Sho Akamine (Max Planck Institute for Psycholinguistics)

SWiFT-MLP is a hands-on, **one-day** workshop on practical, reproducible workflows for analysing multimodal
communication. It is designed for researchers and students working with audio/video data who want to use new
technology to facilitate speech and gesture annotation and to analyse gesture kinematics.

Building on the earlier MEDAL workshop on automatic processing of multimodal interaction, SWiFT-MLP walks
through a complete semi-automated pipeline — from raw video to a multimodal analysis — in four tutorials:

- automatically extract body keypoints using **MediaPipe**
- automatically segment manual **gestures**
- automatically transcribe speech using **WhisperX** and export to **ELAN**
- compare signals across modalities (speech, gesture) and analyse **gesture representations**

The workshop is especially well suited to the MMSYM community, bringing together perspectives from multimodal
communication, linguistics, NLP, computer vision, and cognitive science. It aims to support researchers who need
robust, transparent, and reusable workflows for studying embodied communication in interaction.

---

## Schedule

One day, four tutorials: **two in the morning, two in the afternoon**.
Full schedule: <https://swift-mlp.github.io/2026/#schedule>

| Time | Session | Tutorial |
|------|---------|----------|
| 09:30–09:45 | Welcome and introduction | — |
| 09:45–10:45 | Setup of the environment (VS Code, Miniconda) | — |
| 10:45–11:00 | ☕ Break | |
| **11:00–11:45** | **Pose estimation using MediaPipe** | **Tutorial 1** |
| **11:45–13:00** | **Automatic gesture segmentation** | **Tutorial 2** |
| 13:00–14:00 | 🍽 Lunch | |
| **14:00–15:00** | **Automatic speech transcription using WhisperX** | **Tutorial 3** |
| **15:00–15:45** | **Exporting transcripts into ELAN** | **Tutorial 3** |
| 15:45–16:15 | ☕ Break | |
| **16:15–17:30** | **Multimodal similarity analysis (kinematics + speech)** | **Tutorial 4** |
| 17:30–18:00 | Discussion, wrap-up, and next steps | — |

## The four tutorials

Each tutorial is a self-contained folder with its own notebook(s) and data. They form a pipeline — the output of
one is the input of the next — but each can also be run on its own.

### 🌅 Morning

| Folder | What you do | Out |
|--------|-------------|-----|
| [`Tutorial_1_Pose_Estimation_with_MediaPipe/`](Tutorial_1_Pose_Estimation_with_MediaPipe) | Run MediaPipe over video to track body, hand and face landmarks; inspect and visualise the resulting kinematics | Per-participant keypoint arrays |
| [`Tutorial_2_Gesture_Segmentation/`](Tutorial_2_Gesture_Segmentation) | Turn keypoints into motion features and detect gesture strokes automatically; export the segments for manual checking | Gesture segments (ELAN-ready) |

### 🌆 Afternoon

| Folder | What you do | Out |
|--------|-------------|-----|
| [`Tutorial_3_Speech_Transcription_with_WisperX/`](Tutorial_3_Speech_Transcription_with_WisperX) | Transcribe and force-align speech with WhisperX, then export word- and utterance-level tiers into ELAN (and Praat) | Time-aligned transcripts |
| [`Tutorial_4_Multimodal_Similarity_Analysis/`](Tutorial_4_Multimodal_Similarity_Analysis) | Encode gestures with a self-supervised skeleton model and speech with Dutch BERT; compare gestures within and across speakers and dialogues | Gesture/speech embeddings + similarity analyses |

Tutorial 4 has its own [README](Tutorial_4_Multimodal_Similarity_Analysis/README.md) with details on the data and
the pretrained model.

## Prerequisites

- **Level**: suitable for anyone interested in multimodal interaction.
- **Skills**: basic programming ability, preferably in [Python](https://www.python.org/).
- **Hardware**: a laptop is enough; a GPU is optional (everything runs on CPU, just slower).

## Software

We use the tools below. Nothing has to be installed in advance — we set everything up in the first session —
but installing VS Code, Miniconda and ELAN beforehand saves time.

- [Visual Studio Code](https://code.visualstudio.com/) (+ the Python and Jupyter extensions)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python environment manager)
- [MediaPipe](https://mediapipe.dev/) (kinematic feature extraction)
- [WhisperX](https://github.com/m-bain/whisperX) (automated speech transcription & alignment)
- [ELAN](https://archive.mpi.nl/tla/elan) (annotation tool for multimodal data)

## Installation & setup

### 1. Get the materials

```bash
git clone https://github.com/EsamGhaleb/swift-mlp.git
cd swift-mlp
```

(Or download the ZIP from the [repository page](https://github.com/EsamGhaleb/swift-mlp) and unzip it.)

### 2. Create and activate a Conda environment

```bash
conda create --name swift-mlp python=3.10
conda activate swift-mlp
```

Use **Python 3.10** — several packages are not yet compatible with later versions. (The environment name is
yours to choose; earlier materials used `medal`.)

### 3. Install the required packages

From the workshop directory, with the environment active:

```bash
pip install -r requirements.txt
```

### 4. If ffmpeg gives trouble

```bash
conda install -c conda-forge ffmpeg sox
```

### 5. Extra packages for Tutorial 4 (only if you re-read the raw annotations)

```bash
python -m spacy download nl_core_news_sm
```

The tutorial ships a pre-computed table of embeddings, so these are only needed if you rebuild it from the ELAN
files and transcripts. See the [Tutorial 4 README](Tutorial_4_Multimodal_Similarity_Analysis/README.md).

## Running the notebooks

Open the workshop folder in Visual Studio Code, then open the notebook inside the tutorial folder for the current
session and select the `swift-mlp` environment as the kernel (top right of the notebook). Run the cells from top
to bottom; every notebook is meant to be read as much as executed.

## Background reading

- Ghaleb, E., Burenko, I., Rasenberg, M., Pouw, W., Toni, I., Uhrig, P., Wilson, A., Holler, J., Özyürek, A., &
  Fernández, R. (2024). *Learning Co-Speech Gesture Representations in Dialogue through Contrastive Learning: An
  Intrinsic Evaluation.* ICMI '24. <https://doi.org/10.1145/3678957.3685707>
- Ghaleb, E., et al. (2025). *I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal
  Dialogue.*
- Akamine, S., Meyer, A. S., & Özyürek, A. *Lexical alignment and speaker visibility influence gestural alignment
  in conversation* (manuscript) — the video-mediated corpus used in Tutorial 4.

## Contact

Questions, or something not working? Get in touch:

- Esam Ghaleb — [esamghaleb.github.io](https://esamghaleb.github.io/)
- Sho Akamine — Max Planck Institute for Psycholinguistics
