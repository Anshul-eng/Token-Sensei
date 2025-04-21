import pandas as pd
from collections import Counter

# MeCab (via Fugashi)
from fugashi import Tagger as MeCabTagger

# Sudachi
from sudachipy import tokenizer, dictionary

# SentencePiece
#import sentencepiece as spm

# Optional: Juman++
import subprocess

# === Functions for Tokenization ===

def tokenize_mecab(text):
    tagger = MeCabTagger()
    tokens = [word.surface for word in tagger(text)]
    return tokens

def tokenize_sudachi(text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    tokens = [m.surface() for m in tokenizer_obj.tokenize(text, mode)]
    return tokens

# def tokenize_sentencepiece(text, model_path='spm.model'):
#     sp = spm.SentencePieceProcessor()
#     sp.load(model_path)
#     tokens = sp.encode(text, out_type=str)
#     return tokens

def tokenize_jumanpp(text):
    result = subprocess.run(['jumanpp'], input=text, text=True, stdout=subprocess.PIPE)
    tokens = []
    for line in result.stdout.splitlines():
        if line == 'EOS':
            break
        parts = line.split(' ')
        if parts:
            tokens.append(parts[0])
    return tokens

# === Sample Texts ===
japanese_texts = [
    "これは日本語の文章です。",
    "明日は晴れるでしょう。",
    "TF-IDFは、Term Frequency-Inverse Document Frequency（用語頻度-逆文書頻度）の略です。これは、コレクションまたはコーパス内の文書における単語の重要度を反映する数値統計です。テキストマイニング、情報検索、自然言語処理（NLP）などのタスクでよく使用されます。"
]

# === Run Tokenizers ===
results = []
for text in japanese_texts:
    for name, tokenizer_fn in {
        'MeCab': tokenize_mecab,
        'SudachiPy': tokenize_sudachi,
        # Uncomment the line below if SentencePiece model is available
        # 'SentencePiece': lambda t: tokenize_sentencepiece(t, model_path='your_model.model'),
        # Uncomment the line below if Juman++ is installed
        # 'Juman++': tokenize_jumanpp,
    }.items():
        tokens = tokenizer_fn(text)
        results.append({
            'input_text': text,
            'algorithm': name,
            'tokens': tokens,
            'total_token_count': len(tokens),
            'unique_token_count': len(set(tokens))
        })

# === Save to CSV ===
df = pd.DataFrame(results)
df.to_csv("japanese_tokenization_results.csv", index=False, encoding="utf-8-sig")
print("✅ CSV saved as 'japanese_tokenization_results.csv'")
