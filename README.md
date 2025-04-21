# 🇯🇵 Japanese Tokenization Research

This project compares multiple Japanese tokenization algorithms:  
- **MeCab** (via [Fugashi](https://github.com/polm/fugashi))  
- **SudachiPy**  
- **SentencePiece** (using a pretrained model)  
- **Juman++** (optional)

The script runs all available tokenizers on given Japanese texts and exports a comparison CSV with token counts.

---

## 📦 Requirements

Install Python dependencies:

```bash
pip install pandas fugashi sudachipy sentencepiece

**Project Structure**
tokenization-research/
│
├── tokenization_compare.py    # Main script
├── spm.model                  # Pretrained SentencePiece model (downloaded)
├── japanese_tokenization_results.csv  # Output file
└── README.md
