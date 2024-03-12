import os
import sentencepiece as spm


vocab_file = r"C:\Users\cbigo\Desktop\temp\kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)