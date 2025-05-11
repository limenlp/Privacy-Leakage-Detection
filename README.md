# Privacy-Leakage-Detection

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

## 📦 Acknowledgement

The `main_framework` component of this project used the open-source repository [Recognize Anything](https://github.com/xinyu1205/recognize-anything) by [@xinyu1205](https://github.com/xinyu1205).  
We gratefully acknowledge their contributions and reuse of their work in our privacy leakage detection pipeline.

Please refer to the original repository for full implementation details and license terms.
