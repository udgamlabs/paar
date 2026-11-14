"""Translate English into Gujarati, Punjabi, Assamese or Odia from the command line, offline after the first run.

    pip install ctranslate2 sentencepiece huggingface_hub
    python translate.py gu "The train leaves at 6:45 tomorrow morning."
    echo "Can you send me the invoice?" | python translate.py or

Downloads the 17.7 MB int8 model on first use (cached by huggingface_hub), splits the text into sentences, and
translates each one on its own, as the models were trained.
"""
import re
import sys

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

REPO = "udgamlabs/mt-draft-en-{lang}"
SPLIT = re.compile(r"(?<=[.!?])\s+")


def load(lang):
    path = snapshot_download(REPO.format(lang=lang), allow_patterns=["config.json", "ct2-int8/*"])
    sp = spm.SentencePieceProcessor(model_file=f"{path}/ct2-int8/source.spm")
    return sp, ctranslate2.Translator(f"{path}/ct2-int8", device="cpu", compute_type="int8")


def translate(sp, tr, text):
    lines = []
    for para in text.strip().split("\n"):
        sents = [s for s in SPLIT.split(para.strip()) if s]
        if not sents:
            lines.append("")
            continue
        pieces = [p[:127] + ["</s>"] for p in sp.encode(sents, out_type=str)]
        out = tr.translate_batch(pieces, beam_size=1, max_decoding_length=200)
        lines.append(" ".join(sp.decode([t for t in r.hypotheses[0] if t != "</s>"]) for r in out))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gu", "pa", "as", "or"):
        sys.exit("usage: translate.py gu|pa|as|or [English text]   (reads stdin when no text is given)")
    sp, tr = load(sys.argv[1])
    print(translate(sp, tr, " ".join(sys.argv[2:]) or sys.stdin.read()))
