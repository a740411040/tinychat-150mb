"""Prepare a bounded, public Chinese pilot corpus and provisional tokenizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import time
import unicodedata

import numpy as np
import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "epfml/FineWeb2-HQ"
LANGUAGE = "cmn_Hani"
SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>",
            "<|user|>", "<|assistant|>", "<|end_turn|>"]


def clean_text(value: str) -> str | None:
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").strip()
    if not 200 <= len(text) <= 100_000 or "\ufffd" in text:
        return None
    visible = sum(not ch.isspace() for ch in text)
    chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    if chinese / max(visible, 1) < 0.25:
        return None
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", "[IP]", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def prepare(out: Path, max_mib: int, max_seconds: int, seed: int) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing data directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    api = HfApi(token=False)
    revision = api.dataset_info(REPO).sha
    files = api.list_repo_tree(REPO, repo_type="dataset", revision=revision,
                               path_in_repo=LANGUAGE, recursive=False)
    first_shard = next(item.path for item in files if item.path.endswith(".parquet"))
    data_url = f"hf://datasets/{REPO}@{revision}/{first_shard}"
    source = {"repo": REPO, "revision": revision, "config": LANGUAGE,
              "file": first_shard, "columns": ["text"], "license": "ODC-By-1.0",
              "source_url": f"https://huggingface.co/datasets/{REPO}",
              "usage_terms": "Also subject to Common Crawl terms",
              "sampling": "bounded prefix of first Chinese shard; not representative",
              "provisional_tokenizer": True}
    (out / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    # Column projection avoids fetching the corpus' embedding columns.
    source_handle = fsspec.open(data_url, mode="rb", token=False, block_size=4*2**20).open()
    parquet = pq.ParquetFile(source_handle)
    stream = (row for batch in parquet.iter_batches(batch_size=256, columns=["text"])
              for row in batch.to_pylist())
    docs: list[str] = []
    seen: set[str] = set()
    text_bytes = scanned = 0
    last_log = time.monotonic()
    limit = max_mib * 1024 * 1024
    for row in stream:
        scanned += 1
        if time.monotonic() - start > max_seconds:
            break
        text = clean_text(row["text"])
        if text is None or text in seen:
            continue
        size = len(text.encode("utf-8"))
        if text_bytes + size > limit:
            break
        docs.append(text)
        seen.add(text)
        text_bytes += size
        if time.monotonic() - last_log >= 10:
            print(json.dumps({"phase": "download", "documents": len(docs),
                              "text_mib": round(text_bytes / 2**20, 2)},
                             ensure_ascii=False), flush=True)
            last_log = time.monotonic()
    source_handle.close()
    if len(docs) < 300 or text_bytes < 2**20:
        raise RuntimeError(f"Insufficient bounded corpus: {len(docs)} docs, {text_bytes} bytes")
    random.Random(seed).shuffle(docs)
    n_holdout = max(1, len(docs) // 100)
    splits = {"train": docs[2*n_holdout:], "validation": docs[:n_holdout],
              "test": docs[n_holdout:2*n_holdout]}
    for name, texts in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=16384, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=SPECIALS, show_progress=False)
    print("Training provisional ByteLevel BPE on the training split", flush=True)
    tokenizer.train_from_iterator(splits["train"], trainer=trainer,
                                  length=len(splits["train"]))
    if tokenizer.get_vocab_size() != 16384:
        raise RuntimeError(f"Vocabulary too small: {tokenizer.get_vocab_size()}")
    probe = "中文训练，数字123；English🙂以及生僻字龘。\n第二行。"
    if tokenizer.decode(tokenizer.encode(probe).ids) != probe:
        raise RuntimeError("Tokenizer round-trip failed")
    tokenizer.save(str(out / "tokenizer.json"))
    eos = tokenizer.token_to_id("<|eos|>")
    counts = {}
    for name, texts in splits.items():
        count = 0
        with (out / f"{name}.bin").open("wb") as handle:
            for offset in range(0, len(texts), 64):
                encoded = tokenizer.encode_batch(texts[offset:offset+64])
                for item in encoded:
                    ids = np.asarray(item.ids + [eos], dtype=np.uint16)
                    ids.tofile(handle)
                    count += ids.size
        counts[name] = {"documents": len(texts), "tokens": count,
                        "binary_bytes": count * 2}
    meta = {**source, "seed": seed, "raw_text_bytes": text_bytes,
            "rows_scanned": scanned, "splits": counts, "dtype": "uint16",
            "byteorder": "little", "vocab_size": 16384,
            "tokenizer_path": str((out / "tokenizer.json").resolve()),
            "special_token_ids": {name: tokenizer.token_to_id(name) for name in SPECIALS},
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "limitations": ["small prefix sample", "exact dedup only",
                             "provisional tokenizer; unsuitable as a frozen production corpus"]}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pilot"))
    parser.add_argument("--max-mib", type=int, default=32, choices=range(4, 65))
    parser.add_argument("--max-seconds", type=int, default=300, choices=range(30, 601))
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    prepare(args.output, args.max_mib, args.max_seconds, args.seed)


if __name__ == "__main__":
    main()
