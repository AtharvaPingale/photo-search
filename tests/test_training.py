from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import torch

from api.db.session import get_conn
from eval import dataset
from tests.conftest import make_photo


@pytest.fixture(scope="module")
def tiny_clip():
    import open_clip

    torch.manual_seed(0)
    # random init, no download: architecture is all these tests need
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    return model.eval(), open_clip.get_tokenizer("ViT-B-32")


def test_lora_merge_is_exact(tiny_clip):
    from training.finetune_clip import add_lora, merge_lora

    model, tok = tiny_clip
    import copy

    m = copy.deepcopy(model)
    params = add_lora(m, rank=4, alpha=8)
    assert params and all(p.requires_grad for p in params)
    assert not any(p.requires_grad for n, p in m.named_parameters() if "parametrizations" not in n)
    with torch.no_grad():
        for p in params[1::2]:  # B matrices start at zero; make the adapters do something
            p.normal_(0, 0.02)
    toks = tok(["a dog on a beach", "a red car"])
    img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        t1, i1 = m.encode_text(toks), m.encode_image(img)
        base = model.encode_text(toks)
    assert not torch.allclose(t1, base, atol=1e-4)  # adapters changed the function
    merge_lora(m)
    with torch.no_grad():
        t2, i2 = m.encode_text(toks), m.encode_image(img)
    assert torch.allclose(t1, t2, atol=1e-4) and torch.allclose(i1, i2, atol=1e-4)
    assert set(m.state_dict()) == set(model.state_dict())  # loads into a plain CLIP


def test_unfreeze_last_layers(tiny_clip):
    import copy

    from training.finetune_clip import unfreeze_last

    m = copy.deepcopy(tiny_clip[0])
    params = unfreeze_last(m, 2)
    n = sum(p.numel() for p in params)
    total = sum(p.numel() for p in m.parameters())
    assert 0.05 < n / total < 0.4
    assert not m.visual.transformer.resblocks[0].attn.in_proj_weight.requires_grad
    assert m.visual.transformer.resblocks[-1].attn.in_proj_weight.requires_grad


def test_clip_loss_masks_duplicate_texts():
    from training.finetune_clip import clip_loss

    img = torch.eye(3)
    txt = torch.eye(3)
    scale = torch.tensor(0.0)
    a = clip_loss(img, txt, scale, ["a", "b", "c"])
    txt2 = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    # rows 0 and 1 share a text: without masking they'd be each other's hard negatives
    b = clip_loss(torch.stack([txt2[0], txt2[1], txt2[2]]), txt2, scale, ["same", "same", "c"])
    assert torch.isfinite(a) and torch.isfinite(b)


def test_build_pairs_excludes_eval_photos_and_eval_like_feedback(indexed, tmp_path, monkeypatch):
    from training import build_pairs
    from workers import caption

    # a burst neighbour of an eval photo, 5 seconds later on the same camera
    make_photo(indexed / "2025/chicago/red_car_2.jpg", (221, 21, 21), gps=(41.88, -87.63),
               taken=datetime(2025, 6, 1, 18, 30, 5))  # fmt: skip
    from workers.embed import embed_pending
    from workers.ingest import scan

    scan([indexed])
    embed_pending()
    caption.set_captioner(caption.FakeCaptioner())
    try:
        caption.caption_pending()
    finally:
        caption.set_captioner(None)
    with get_conn() as conn:
        rows = {
            Path(r["path"]).name: r for r in conn.execute("SELECT id, path, file_hash FROM photos")
        }
        conn.execute(
            "INSERT INTO feedback (query, photo_id, label) VALUES ('red car', %s, 1)",
            (rows["red_sign.jpg"]["id"],),
        )
        conn.execute(
            "INSERT INTO feedback (query, photo_id, label) VALUES ('neon sign at dusk', %s, 1)",
            (rows["red_sign.jpg"]["id"],),
        )
        conn.commit()
    qfile = tmp_path / "q.jsonl"
    monkeypatch.setattr(dataset, "QUERIES_FILE", qfile)
    dataset.save_queries(
        [
            dataset.EvalQuery(
                id="q0001",
                query="red car",
                split="test",
                relevant=[rows["red_car.jpg"]["file_hash"]],
            )
        ],
        qfile,
    )
    man = build_pairs.build(tmp_path / "train", val_frac=0.0)
    assert man["leakage_exclusions"]["eval_photos"] == 1
    assert man["leakage_exclusions"]["same_scene_neighbours"] == 1  # red_car_2
    assert man["dropped"]["feedback_dropped_eval_like"] == 1  # "red car" is an eval query
    train = [
        json.loads(line) for line in (tmp_path / "train" / "train.jsonl").read_text().splitlines()
    ]
    used = {r["photo_id"] for r in train}
    assert (
        str(rows["red_car.jpg"]["id"]) not in used and str(rows["red_car_2.jpg"]["id"]) not in used
    )
    assert any(r["text"] == "neon sign at dusk" for r in train)
    assert build_pairs.caption_sentences(
        "The image shows a red car parked on a street. It is sunny. Ok"
    ) == [
        "a red car parked on a street.",
        "It is sunny.",
    ]
