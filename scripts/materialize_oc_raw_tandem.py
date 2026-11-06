#!/usr/bin/env python3
"""Clone the tandem-OC suite onto the raw JAXAtari observation meta head.

Skills stay ego. The meta network is ``flat_mlp`` with ``META_INPUT_MODE: flat``,
which flattens the native object observation and scores one Q per target slot.
This is not ``oc_simple_rel_role``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/experiment/paper_eval/thesis_final"
DST = ROOT / "configs/experiment/paper_eval/oc_raw_tandem"


def rewrite(text: str) -> str:
    text = text.replace("tandem_oc", "tandem_ocraw")
    text = text.replace("per_object_mlp", "flat_mlp")
    text = text.replace("oc_simple_rel_role", "flat")
    text = text.replace("thesis_final/", "oc_raw_tandem/")
    text = text.replace("suite:thesis_final", "suite:oc_raw_tandem")
    text = text.replace(
        "# Thesis-final tandem oc meta",
        "# Raw JAXAtari-OC tandem meta",
    )
    text = text.replace(
        "# Thesis-final home mod eval",
        "# Raw JAXAtari-OC tandem home eval",
    )
    text = text.replace(
        "# Thesis-final ZS",
        "# Raw JAXAtari-OC tandem skill transfer",
    )
    return text


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC.glob("*tandem_oc*.yaml"))
    if len(sources) != 60:
        raise SystemExit(f"expected 60 tandem_oc configs, found {len(sources)}")
    for path in sources:
        name = path.name.replace("tandem_oc", "tandem_ocraw")
        (DST / name).write_text(rewrite(path.read_text()))
    print(f"wrote {len(sources)} configs to {DST}")


if __name__ == "__main__":
    main()
