![Draft · Udgam Labs](./assets/banner.png)

**UDGAM LABS** · Translator · English → 12 languages · 17.7 MB each at int8

# Draft

Draft is twelve small translation models, one per direction: English → Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati,
Urdu, Kannada, Malayalam, Punjabi, Odia and Assamese. Each has 17.3M parameters and is 17.7 MB at int8, and runs
offline on an ordinary CPU (about 30 ms a sentence on one core of a Mac mini) or in a web browser with
transformers.js. They were distilled from IndicTrans2. Each is about 8.5 times smaller to download than indicMT;
by language, it is level with indicMT or up to 4.8 chrF++ behind it. Against Firefox's offline models: English → Gujarati is ahead; Bengali and
Telugu are ahead on everyday conversation and behind on news-style text; the others are behind by up to 3.6; Firefox
has no Punjabi, Odia or Assamese model.

[Model cards on Hugging Face](https://huggingface.co/udgamlabs/mt-draft-en-gu) · [Try it](https://huggingface.co/spaces/udgamlabs/mt-draft) · [udgamlabs.com](https://udgamlabs.com)

## Get started

```bash
pip install ctranslate2 sentencepiece huggingface_hub
python examples/translate.py gu "Can you send me the invoice for the last three months?"
echo "The train leaves at 6:45 tomorrow morning." | python examples/translate.py or
```

The first run downloads the 17.7 MB model for that language; after that it works offline. `hi`, `bn`, `mr`, `te`,
`ta`, `gu`, `ur`, `kn`, `ml`, `pa`, `or` and `as` pick the language. In your own code, see the three quick starts
(Transformers, CTranslate2 and transformers.js for JavaScript) on each model card.

## In the browser

Each model repo has an ONNX build for [transformers.js](https://huggingface.co/docs/transformers.js), about
29 MB per language at int8, downloaded once and then run on the visitor's device:

```js
import { pipeline } from "@huggingface/transformers";
const translate = await pipeline("translation", "udgamlabs/mt-draft-en-hi", { dtype: "q8" });
console.log((await translate("Can you send me the invoice?"))[0].translation_text);
```

Tested in Chrome and Safari on a Mac: about 40 ms a sentence, the same outputs as the Python quick start in fp32 and
the same quality in int8. Phones and other browsers are not measured.

## Fine-tune it on your own sentences

```bash
pip install -r finetune/requirements.txt
python finetune/finetune.py --lang gu --train pairs.tsv --dev dev.tsv --out my-model --ct2
```

`pairs.tsv` holds one English sentence, a tab, and its translation per line. The output is a normal Transformers model
with a card whose header sets `base_model`, so a published copy links back; `--ct2` also writes the int8 build. Details
and settings: [`finetune/README.md`](finetune/README.md).

## How it works

Each model is a Marian encoder-decoder (6 encoder and 2 decoder layers, width 256, one shared 32,000-piece
SentencePiece vocabulary), trained from random weights on English sentences from IndicCorp v2 that IndicTrans2
dist-200M had translated. It translates one sentence at a time with greedy decoding; `examples/translate.py` splits
text into sentences first.

## Results

chrF++ (higher is better), FLORES+ devtest / IN22-Conv, our harness, the same sentences for every system; greedy for
ours and Firefox's offline models, beam 5 for indicMT (153.8M, 169.2 MB at int8). Full tables with paired intervals are
on each model card.

| Direction | Hugging Face | Ours | indicMT | Firefox offline (size) | Ours at int8 |
|---|---|---:|---:|---:|---:|
| English → Hindi | [`udgamlabs/mt-draft-en-hi`](https://huggingface.co/udgamlabs/mt-draft-en-hi) | 55.49 / 44.92 | 56.92 / 47.22 | 58.13 / 46.79 (17.1 MB) | 19.6 MB |
| English → Bengali | [`udgamlabs/mt-draft-en-bn`](https://huggingface.co/udgamlabs/mt-draft-en-bn) | 50.41 / 44.88 | 50.99 / 47.21 | 51.34 / 43.57 (17.1 MB) | 19.8 MB |
| English → Marathi | [`udgamlabs/mt-draft-en-mr`](https://huggingface.co/udgamlabs/mt-draft-en-mr) | 48.02 / 44.55 | 48.68 / 46.07 | 51.07 / 47.70 (31.6 MB) | 19.8 MB |
| English → Telugu | [`udgamlabs/mt-draft-en-te`](https://huggingface.co/udgamlabs/mt-draft-en-te) | 55.50 / 40.94 | 56.56 / 42.62 | 56.62 / 40.21 (17.1 MB) | 19.9 MB |
| English → Tamil | [`udgamlabs/mt-draft-en-ta`](https://huggingface.co/udgamlabs/mt-draft-en-ta) | 53.86 / 36.12 | 53.95 / 38.14 | 56.67 / 37.11 (31.6 MB) | 20.0 MB |
| English → Gujarati | [`udgamlabs/mt-draft-en-gu`](https://huggingface.co/udgamlabs/mt-draft-en-gu) | 53.21 / 49.60 | 53.95 / 50.12 | 52.55 / 47.49 (17.1 MB) | 19.7 MB |
| English → Urdu | [`udgamlabs/mt-draft-en-ur`](https://huggingface.co/udgamlabs/mt-draft-en-ur) | 48.24 / 56.06 | 49.15 / 57.91 | 50.68 / 59.62 (31.6 MB) | 19.4 MB |
| English → Kannada | [`udgamlabs/mt-draft-en-kn`](https://huggingface.co/udgamlabs/mt-draft-en-kn) | 51.93 / 30.29 | 51.65 / 31.66 | 53.14 / 30.64 (17.1 MB) | 19.9 MB |
| English → Malayalam | [`udgamlabs/mt-draft-en-ml`](https://huggingface.co/udgamlabs/mt-draft-en-ml) | 52.64 / 35.59 | 52.22 / 38.22 | 53.38 / 35.72 (17.1 MB) | 20.0 MB |
| English → Punjabi | [`udgamlabs/mt-draft-en-pa`](https://huggingface.co/udgamlabs/mt-draft-en-pa) | 49.41 / 51.47 | 51.89 / 54.72 | none | 19.5 MB |
| English → Odia | [`udgamlabs/mt-draft-en-or`](https://huggingface.co/udgamlabs/mt-draft-en-or) | 46.40 / 36.37 | 50.24 / 41.19 | none | 19.7 MB |
| English → Assamese | [`udgamlabs/mt-draft-en-as`](https://huggingface.co/udgamlabs/mt-draft-en-as) | 39.90 / 42.57 | 41.42 / 44.70 | none | 19.7 MB |

## Where it falls short

- Behind IndicTrans2 dist-200M, their teacher, and behind indicMT on every set except FLORES+ for Kannada,
  Malayalam and Tamil (level), by up to 4.8 chrF++.
- Behind Firefox's offline models for Hindi, Marathi, Tamil, Urdu, Kannada and Malayalam, and for Bengali and Telugu
  on news-style text (0.3 to 3.6 chrF++).
- Some letters are missing from each vocabulary and come out as "⁇": a few independent vowels per script (for example
  Gujarati ઔ, Bengali ঔ, Malayalam ഋ), native digits in several scripts (0-9 are written instead), most Urdu
  short-vowel marks. 0.2 to 4% of test outputs carry one; each model card lists its own.
- Names are often mangled; idioms are translated word for word; long formal sentences lose meaning more often.
- English into these twelve only. No native-speaker review yet. Speed is measured on one Mac mini (CPU, Chrome,
  Safari), not on phones.

## Licence

Apache-2.0 for the code and the model weights. See `NOTICE` for IndicTrans2 (MIT) and IndicCorp v2 (CC0).

---

**About Udgam Labs.** Udgam Labs is an independent AI research lab. We work from first principles: before we build, we understand why things work, whether we are training a model, fine-tuning one or applying AI to a new problem. Udgam (उद्गम) is Sanskrit for origin. [udgamlabs.com](https://udgamlabs.com)
