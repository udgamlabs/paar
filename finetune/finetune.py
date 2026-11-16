"""Fine-tune one of the translators on your own English → target sentence pairs (Transformers Seq2SeqTrainer).

    pip install -r requirements.txt
    python finetune.py --lang gu --train pairs.tsv --out my-model [--dev dev.tsv] [--epochs 3]

pairs.tsv: one pair per line, English, a tab, then the translation. A few thousand good pairs in your domain (product
names, a support desk's phrasing, a style guide) is a sensible start. The output folder is a normal Transformers model
with the tokenizer, plus a README.md whose header sets base_model, so a published copy links back to the original.
Convert it to CTranslate2 int8 afterwards with --ct2 (see README.md in this folder for why one setting is needed).
"""
import argparse
import csv
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (DataCollatorForSeq2Seq, MarianMTModel, MarianTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

REPO = "udgamlabs/Paar-en-{lang}"


def read_pairs(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 2 and r[0] and r[1]]
    return Dataset.from_dict({"src": [r[0] for r in rows], "tgt": [r[1] for r in rows]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hi", "bn", "mr", "te", "ta", "gu", "ur", "kn", "ml", "pa", "or", "as"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--ct2", action="store_true", help="also write an int8 CTranslate2 folder to OUT/ct2-int8")
    a = ap.parse_args()

    repo = REPO.format(lang=a.lang)
    tok = MarianTokenizer.from_pretrained(repo)
    model = MarianMTModel.from_pretrained(repo)

    def enc(b):
        x = tok(b["src"], truncation=True, max_length=128)
        x["labels"] = tok(text_target=b["tgt"], truncation=True, max_length=200)["input_ids"]
        return x

    train = read_pairs(a.train).map(enc, batched=True, remove_columns=["src", "tgt"])
    dev = read_pairs(a.dev).map(enc, batched=True, remove_columns=["src", "tgt"]) if a.dev else None
    steps = a.max_steps if a.max_steps > 0 else math.ceil(a.epochs * math.ceil(len(train) / a.batch))
    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(a.out) / "checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        learning_rate=a.lr, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        warmup_steps=max(1, steps // 20), weight_decay=0.01, label_smoothing_factor=0.1, logging_steps=50,
        eval_strategy="epoch" if dev is not None else "no", save_strategy="epoch", save_total_limit=2,
        predict_with_generate=False, report_to=[], use_cpu=not (torch.cuda.is_available()))
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model), processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    Path(a.out, "README.md").write_text(
        f"---\nbase_model: {repo}\nbase_model_relation: finetune\nlibrary_name: transformers\n"
        f"pipeline_tag: translation\nlanguage: [en, {a.lang}]\n---\n\n"
        f"Fine-tuned from [{repo}](https://huggingface.co/{repo}) with finetune.py.\n", encoding="utf-8")
    if a.ct2:
        to_ct2(a.out)
    print("saved", a.out)


def to_ct2(path):
    """CTranslate2's Marian converter assumes the original Marian layout: <pad> is the last vocabulary row and the
    decoder starts from an all-zero embedding. These models keep <pad> at id 2 and start the decoder from its learned
    row, so nothing is dropped and the decoder starts from <pad> itself. Without the second fix the int8 build drifts
    from the fp32 model (for one language it lost 2 chrF++)."""
    import ctranslate2
    from ctranslate2.converters import transformers as ct2t

    ct2t.MarianMTLoader._remove_pad_weights = lambda self, spec: None
    ct2t.MarianMTLoader.set_decoder = lambda self, spec, dec: ct2t.BartLoader.set_decoder(self, spec, dec)
    set_config = ct2t.MarianMTLoader.set_config

    def _set_config(self, config, model, tokenizer):
        set_config(self, config, model, tokenizer)
        config.decoder_start_token = tokenizer.pad_token

    ct2t.MarianMTLoader.set_config = _set_config
    ctranslate2.converters.TransformersConverter(path).convert(str(Path(path) / "ct2-int8"), quantization="int8",
                                                               force=True)
    import shutil
    shutil.copy(Path(path) / "source.spm", Path(path) / "ct2-int8" / "source.spm")


if __name__ == "__main__":
    main()
