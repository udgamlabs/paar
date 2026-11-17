"""Translate English into one of twelve languages from the command line, offline after the first run.

    pip install ctranslate2 sentencepiece huggingface_hub
    python translate.py gu "The train leaves at 6:45 tomorrow morning."
    echo "Can you send me the invoice?" | python translate.py or

Downloads the 17.7 MB int8 model on first use (cached by huggingface_hub), splits the text into sentences, and
translates each one on its own, as the models were trained.
"""
import re
import sys
import unicodedata

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

REPO = "udgamlabs/Paar-en-{lang}"
SPLIT = re.compile(r"(?<=[.!?])\s+")


def load(lang):
    path = snapshot_download(REPO.format(lang=lang), allow_patterns=["config.json", "ct2-int8/*"])
    sp = spm.SentencePieceProcessor(model_file=f"{path}/ct2-int8/source.spm")
    return sp, ctranslate2.Translator(f"{path}/ct2-int8", device="cpu", compute_type="int8")


# Assamese and Odia write some words with an apostrophe inside (হ'ব, କ'ଣ); the models write a one-letter word, a space
# and a quote instead (হ "ব, also ক" ত). This joins them back. Measured: Assamese +0.09 / +0.34 chrF++ (FLORES+ /
# IN22-Conv), Odia -0.01 / +0.18; the other ten languages are better without it.
_SPLIT = re.compile(r"""(^|\s)([^\W\d_][^\w\s]*?)(?: ["'] ?|["'] )(?=[^\W\d_])""")


def join_apostrophes(text):
    one = lambda w: unicodedata.category(w[0])[0] == "L" and all(unicodedata.category(c)[0] == "M" for c in w[1:])
    return _SPLIT.sub(lambda m: m.group(1) + m.group(2) + "'" if one(m.group(2)) else m.group(0), text)


def translate(sp, tr, text, lang=None):
    lines = []
    for para in text.strip().split("\n"):
        sents = [s for s in SPLIT.split(para.strip()) if s]
        if not sents:
            lines.append("")
            continue
        pieces = [p[:127] + ["</s>"] for p in sp.encode(sents, out_type=str)]
        out = tr.translate_batch(pieces, beam_size=1, max_decoding_length=200)
        line = " ".join(sp.decode([t for t in r.hypotheses[0] if t != "</s>"]) for r in out)
        lines.append(join_apostrophes(line) if lang in ("as", "or") else line)
    return "\n".join(lines)


if __name__ == "__main__":
    LANGS = ("hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "ml", "pa", "or", "as")
    if len(sys.argv) < 2 or sys.argv[1] not in LANGS:
        sys.exit("usage: translate.py " + "|".join(LANGS) + " [English text]   (reads stdin when no text is given)")
    sp, tr = load(sys.argv[1])
    print(translate(sp, tr, " ".join(sys.argv[2:]) or sys.stdin.read(), sys.argv[1]))
