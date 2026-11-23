import json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=False)
revision = "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e"
entries = sorted((item for item in api.list_repo_tree(
    "epfml/FineWeb2-HQ", repo_type="dataset", revision=revision,
    path_in_repo="cmn_Hani", recursive=False) if item.path.endswith(".parquet")),
    key=lambda item: item.path)
selected = [entries[min(len(entries)-1, int((i+0.5)*len(entries)/8))] for i in range(8)]
result = {"pretrain": {"repo": "epfml/FineWeb2-HQ", "revision": revision,
                       "available_chinese_shards": len(entries),
                       "selected": [{"path": x.path, "bytes": x.size} for x in selected]},
          "sft": []}
for repo in ("BelleGroup/train_0.5M_CN", "BelleGroup/multiturn_chat_0.8M"):
    info = api.dataset_info(repo, files_metadata=True)
    files = [{"path": x.rfilename, "bytes": x.size} for x in info.siblings
             if x.rfilename.endswith((".json", ".jsonl", ".parquet"))]
    result["sft"].append({"repo": repo, "revision": info.sha, "files": files[:10],
                          "usage": "research only; no commercial derivatives per publisher"})
Path("planning/train_150mb/stage1_sources.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
