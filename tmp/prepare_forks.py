import json
import shutil
from pathlib import Path

src_root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\open_kernels")
dst_root = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\forks")
if dst_root.exists():
    shutil.rmtree(dst_root)
dst_root.mkdir(parents=True)

forks = [
    {
        "src": "anvithpothula__rsna-base",
        "slug": "rsna-knee-open-rsna-base",
        "title": "RSNA Knee Open rsna-base 0936",
        "origin": "anvithpothula/rsna-base",
    },
    {
        "src": "anhadmahajan06__rsna-knee-take-care-of-your-knee",
        "slug": "rsna-knee-open-take-care",
        "title": "RSNA Knee Open take care 0936",
        "origin": "anhadmahajan06/rsna-knee-take-care-of-your-knee",
    },
    {
        "src": "hyakumanben2025__rsna-knee-dinosaur-v6-repro",
        "slug": "rsna-knee-open-dinosaur-v6",
        "title": "RSNA Knee Open dinosaur v6 0936",
        "origin": "hyakumanben2025/rsna-knee-dinosaur-v6-repro",
    },
    {
        "src": "llccqq624__rsna-knee-dino-protocol-fusion",
        "slug": "rsna-knee-open-protocol-fusion",
        "title": "RSNA Knee Open protocol fusion 0935",
        "origin": "llccqq624/rsna-knee-dino-protocol-fusion",
    },
    {
        "src": "hdhsjdjd__rsna-knee-raptor-quad-w065",
        "slug": "rsna-knee-open-raptor-quad",
        "title": "RSNA Knee Open raptor quad 0935",
        "origin": "hdhsjdjd/rsna-knee-raptor-quad-w065",
    },
]

for item in forks:
    src = src_root / item["src"]
    dst = dst_root / item["slug"]
    dst.mkdir(parents=True)
    meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
    code_file = meta["code_file"]
    shutil.copy2(src / code_file, dst / code_file)
    new_meta = {
        "id": f"dmitriigluzdov/{item['slug']}",
        "title": item["title"],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": meta.get("dataset_sources", []),
        "kernel_sources": meta.get("kernel_sources", []),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": meta.get("model_sources", []),
        "docker_image": meta.get("docker_image", ""),
        "machine_shape": "NvidiaTeslaT4",
    }
    (dst / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    (dst / "ORIGIN.txt").write_text(item["origin"] + "\n", encoding="utf-8")
    print("prepared", item["slug"], "from", item["origin"])
