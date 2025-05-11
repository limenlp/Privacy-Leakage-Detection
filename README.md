# SituationalPriv: A Context-Aware Framework for Privacy Detection and Protection in Vision-Language Models

SituationalPriv, a context-aware framework for privacy leakage detection and protection.
This project provides a privacy leakage detection framework that evaluates whether contextual visual and textual content may lead to unintended privacy disclosure. It includes a unified main framework combining LLM and VLM, as well as LLM-only and VLM-only baselines for comparison. The detection is performed over image-context pairs under both privacy and non-privacy conditions.

---

## 📁 Project Structure

```bash
PRIVACY-LEAKAGE-DETECTION/
│
├── datasets/                          # Input data (images and context)
│   ├── input_images/                  # Raw image files
│   ├── context_privacy.json           # Contexts containing privacy information
│   └── context_nonprivacy.json        # Contexts without privacy risk
│
├── llm_baseline/                      # LLM-only detection pipeline
│   
├── vlm_baseline/                      # VLM-only detection pipeline
│   
├── main_framework/                    # Full pipeline using both LLM and VLM
│
└── README.md                          # Project documentation (this file)
````

## 🚀 Usage

### 0. Environment Setup

Before running the project, you need to set up the appropriate Python environment. We provide a pre-configured environment file named `vlmprivacy4_environment.yml` to simplify the setup process.

#### Create and Activate the Environment

To create a conda environment named `vlmprivacy4` and install all necessary dependencies, run:

```bash
conda env create -f vlmprivacy4_environment.yml
conda activate vlmprivacy4
```

This will install all required packages for the detection framework.

To run the full privacy leakage detection pipeline (which combines LLM and VLM), please ensure:

* The [Recognize Anything](https://github.com/xinyu1205/recognize-anything) repository has been cloned into the following path:

```
main_framework/object_aware_preprocessing/recognize-anything/
```

This module is required for object-aware preprocessing of images.

---

### 1. Main Framework Execution



#### Run a Single Example

To run the pipeline on a single demo image-context pair (e.g., `demo_1`), execute the following:

```bash
cd main_framework
python run_all_steps.py demo_1
```

---

#### Batch Execution

To run multiple image-context pairs in batch mode, use the following command:

```bash
cd main_framework
python autorun.py
```

The script will prompt you to input a **start row** and **end row** corresponding to entries in the `experiments.xlsx` file. This file defines which samples to run and their configurations.

---

### 2. Add and Run New Samples

To evaluate newly added image-context pairs (e.g., `demo_new`), ensure the following:

* Place the new image file at:

  ```
  datasets/input_images/demo_new.jpg
  ```

* Add the corresponding context entry in the following files:

  * If it contains privacy-sensitive content:

    ```
    datasets/context_privacy.json
    ```

  * If it is a non-privacy context:

    ```
    datasets/context_nonprivacy.json
    ```

Make sure the JSON structure matches existing entries, and the `id` field should match the image filename (e.g., `"demo_new"`).



## 📦 Acknowledgement

The `main_framework` component of this project used the open-source repository [Recognize Anything](https://github.com/xinyu1205/recognize-anything) by [@xinyu1205](https://github.com/xinyu1205).  
We gratefully acknowledge their contributions and reuse of their work in our privacy leakage detection pipeline.

Please refer to the original repository for full implementation details and license terms.
