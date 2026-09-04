# MMSYM 2026 Workshop on Semi-Automated Workflows for Facilitating Multimodal Language Processing (SWiFT-MLP) - Esam Ghaleb & Sho Akamine

SWiFT-MLP: Semi-Automated Workflows for Facilitating Multimodal Language Processing is a hands-on workshop on practical, reproducible workflows for analyzing multimodal communication.

It is designed for researchers and students working with audio/video data who want to use new technology to facilitate speech/gesture annotation and analyze gesture kinematics.

Building on the earlier MEDAL workshop on automatic processing of multimodal interaction, SWiFT-MLP introduces semi-automated pipelines to:

   - automatically extract body keypoints using MediaPipe
   - automatically segment manual gestures
   - automatically transcribe speech using WhisperX
   - export annotations to common analysis tools (e.g., Praat, ELAN)
   - compare signals across modalities (e.g., speech, gesture) and analyze gesture kinematics

The workshop is especially well-suited to the MMSYM community, bringing together perspectives from multimodal communication, linguistics, NLP, computer vision, and cognitive science. It aims to support researchers who need robust, transparent, and reusable workflows for studying embodied communication in interaction.
## Prerequisites
- **Level**: Suitable for anyone interested in multimodal interaction.  
- **Skills**: Basic programming ability, preferably in [Python](https://www.python.org/).

## Software
We will use the following tools; no installation required beforehand—we’ll set them up during the workshop:
- [Visual Studio Code](https://code.visualstudio.com/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python environment manager)
- [MediaPipe](https://mediapipe.dev/) (kinematic feature extraction)
- [whisperX](https://github.com/m-bain/whisperX) (automated speech transcription & alignment)
- [ELAN](https://archive.mpi.nl/tla/elan) (annotation tool for multimodal data)

## Schedule: https://swift-mlp.github.io/2026/#schedule

## Download the workshop data and code
You can download the workshop data and code from the GitHub repository: https://github.com/EsamGhaleb/swift-mlp

## Installation & Setup Instructions

Follow these steps to set up the environment and install required packages.

### 1. Install Visual Studio Code and Miniconda

In the first half hour of the workshop, we will set up the environment and install the necessary software. If you want to prepare in advance, you can install the following software:

- **Visual Studio Code**: Download and install from [here](https://code.visualstudio.com/).
   - After installation, install the Python extension for Visual Studio Code.
   - Install the Jupyter extension for running Jupyter notebooks 
- **Miniconda**: Download and install from [here](https://docs.conda.io/en/latest/miniconda.html).
   - Follow the instructions for your operating system to install Miniconda.
   - After installation, open a terminal and run `conda init` to set up your shell for Conda.
- **ELAN**: Download and install from [here](https://archive.mpi.nl/tla/elan). 
   - ELAN is a tool for annotating multimodal data throughout the workshop.

Please note that you do not need to install **MediaPipe** or **whisperX** beforehand; we will install them during the workshop.
  
### 2. Create and activate a Conda environment

```bash
conda create --name medal python=3.10 
```
This command creates a new Conda environment named `medal` with Python version 3.10. Make sure that you use Python 3.10, as some packages may not be compatible with later versions.
After creating the environment, you need to activate it. Run the following command in your terminal:
```bash  
conda activate medal
```
### 3. Install required packages
Go into the workshop directory. You can install the required packages using the provided `requirements.txt` file. This file contains all the necessary dependencies for the workshop.
Make sure you are in the `medal` Conda environment, then run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 4. Reinstall ffmpeg package
If you encounter issues with the `ffmpeg` package, you can reinstall it using Conda. Run the following command in your terminal:
```bash
conda install -c conda-forge ffmpeg
```

## Workshop Materials
You can open the workshop materials in Visual Studio Code. In the interface of Visual Studio Code:

- From the file explorer, open the project folder and select Day 1
  - In the Day 1 folder, you will find the Jupyter notebooks for **Extracting body key points** and **Segmenting Gestures**.
- From the file explorer, open the project folder and select Day 2
  - In the Day 2 folder, you will find the Jupyter notebooks for **Speech Transcription** and **Multimodal Similarity Analysis**.


For any questions or further information, please contact:
- Esam Ghaleb: [esamghaleb.github.io](https://esamghaleb.github.io/)
