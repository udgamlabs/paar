# Fine-tuning

`finetune.py` trains one of the twelve models further on your own English → target sentence pairs with the Transformers
`Seq2SeqTrainer`. Tested on 16 November 2026 with Transformers 5.23 and CTranslate2 4.9 on a Mac mini CPU (a
20-step run on 300 pairs takes a few seconds; real runs want a GPU or patience). After `--ct2`, the int8 build gave
the same translation as the Transformers model for 71% of 200 test sentences (80% in float32); the rest differ by
rounding, not by quality.

```bash
pip install -r requirements.txt
python finetune.py --lang gu --train pairs.tsv --dev dev.tsv --out my-model --epochs 3 --ct2
```

| Option | Default | What it does |
|---|---|---|
| `--lang` | (required) | `hi`, `bn`, `mr`, `te`, `ta`, `gu`, `ur`, `kn`, `ml`, `pa`, `or` or `as`: which model to start from |
| `--train`, `--dev` | | tab-separated files: English, a tab, the translation, one pair per line |
| `--epochs`, `--max-steps` | 3, off | how long to train (`--max-steps` wins when set) |
| `--lr`, `--batch` | 1e-4, 32 | learning rate and batch size |
| `--ct2` | off | also write an int8 CTranslate2 build to `my-model/ct2-int8/` |

**Good data matters more than settings.** A few thousand clean pairs from your own domain (product names, support
replies, a style guide) usually help that domain; a small or noisy set can make general translation worse. Keep a
dev file the model never trains on, and compare it before and after.

**Publishing a fine-tune:** the output's `README.md` already sets `base_model` and `base_model_relation: finetune`,
so Hugging Face links it back to the original. Add what you trained on and how it scores.

**Converting to CTranslate2 yourself:** CTranslate2's Marian converter assumes the original Marian layout, where
`<pad>` is the last vocabulary row and the decoder starts from an all-zero embedding. These models keep `<pad>` at id 2
and start the decoder from its learned row, so `finetune.py` (`to_ct2`) changes two things: it keeps every row, and it
starts the decoder from `<pad>`. Without the first, conversion fails with "Source vocabulary 0 has size 32000 but the
model expected a vocabulary of size 31999"; without the second, the int8 build quietly drifts from the fp32 model.
