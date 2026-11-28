# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The same UTF-8 model metadata written literally or with JSON escapes produces the same model."""
import json
from pathlib import Path
import shutil
import sys
import tempfile

python = Path(__file__).resolve().parents[3] / "python"
sys.path[:0] = [str(python / "convert"), str(python)]
import flyx
from flyx_convert import convert, Reader


def generate(model, library):
    engine = flyx.Engine(model, lib=library, backend="cpu_ref", backend_dir=library.parent,
                         max_ctx=128, speculative=False)
    session = engine.session()
    try:
        return session.generate(text="A small test", max_new_tokens=8, ignore_eos=True)
    finally:
        session.close()
        engine.close()


if __name__ == "__main__":
    assets, library = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="flyx convert ") as tmp:
        directory = Path(tmp)
        model = Path(shutil.copytree(assets / "hf_target_nvfp4", directory / "model"))
        draft = Path(shutil.copytree(assets / "hf_draft", directory / "draft"))
        target, generation, drafter = model / "config.json", model / "generation_config.json", draft / "config.json"
        configs = {p: json.loads(p.read_bytes()) for p in (target, generation, drafter)}
        producer = "Weights · 模型 😁"
        for data in configs.values():
            data["description"] = producer
        configs[generation].update(eos_token_id=[2, 47], temperature=0.6, top_k=0, do_sample=False,
                                   max_length=512, min_p=0.05)
        quant = configs[target].pop("quantization_config")
        quant["producer"]["name"] = producer
        configs[model / "hf_quant_config.json"] = {"quantization": quant}
        outputs = []
        for escaped in (True, False):
            for path, data in configs.items():
                path.write_text(json.dumps(data, ensure_ascii=escaped), encoding="utf-8")
            out = directory / f"model-{escaped}.flyx"
            convert(str(model), str(out), str(draft), log=lambda *_: None)
            outputs.append(out)
        assert outputs[0].read_bytes() == outputs[1].read_bytes()
        with Reader(outputs[1]) as reader:
            assert reader.header["quant"]["producer"] == producer
            assert 47 in reader.header["generation"]["eos_ids"]
            assert reader.header["generation"]["sampling"] == {
                "temperature": 0.6, "top_k": 0, "do_sample": False, "min_p": 0.05}
            assert "max_new_tokens" not in reader.header["generation"]
        expected = generate(outputs[0], library)
        assert len(expected) == 8 and generate(outputs[1], library) == expected
