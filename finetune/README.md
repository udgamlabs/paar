# Fine-tuning

`finetune.py` trains one of the four models further on your own English → target sentence pairs with the Transformers
`Seq2SeqTrainer`. Tested on 14 November 2026 with Transformers 5.23 on a Mac mini CPU (a 20-step run on 300
pairs takes a few seconds; real runs want a GPU or patience).

```bash
pip install -r requirements.txt
python finetune.py --lang gu --train pairs.tsv --dev dev.tsv --out my-model --epochs 3 --ct2
```

| Option | Default | What it does |
|---|---|---|
| `--lang` | (required) | `gu`, `pa`, `as` or `or`: which model to start from |
| `--train`, `--dev` | | tab-separated files: English, a tab, the translation, one pair per line |
| `--epochs`, `--max-steps` | 3, off | how long to train (`--max-steps` wins when set) |
| `--lr`, `--batch` | 1e-4, 32 | learning rate and batch size |
| `--ct2` | off | also write an int8 CTranslate2 build to `my-model/ct2-int8/` |

**Good data matters more than settings.** A few thousand clean pairs from your own domain (product names, support
replies, a style guide) usually help that domain; a small or noisy set can make general translation worse. Keep a
dev file the model never trains on, and compare it before and after.

**Publishing a fine-tune:** the output's `README.md` already sets `base_model` and `base_model_relation: finetune`,
so Hugging Face links it back to the original. Add what you trained on and how it scores.

**Converting to CTranslate2 yourself:** CTranslate2's Marian converter drops the last vocabulary row, assuming it is
`<pad>`. These models keep `<pad>` at id 2, so `finetune.py` turns that step off (`to_ct2`). Converting without it
fails with "Source vocabulary 0 has size 32000 but the model expected a vocabulary of size 31999".
