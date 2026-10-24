"""Local synthetic files exercise guards; no trained model or GPU is used."""
import copy
import csv
import json
from pathlib import Path
import sys
import pytest
from inference.resume import ResumeGuard, inference_identity


def fixture_files(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'synthetic model bytes, not a model')
    (model / 'preprocessor_config.json').write_text('{}')
    for phase in ('a','p','v'):(tmp_path / f'{phase}.png').write_bytes(b'synthetic image')
    rows = [{'metadata':{'logical_key':'x','slice_index':0}, 'messages':[{'role':'user','content':[
        *[{'type':'image','image':f'{phase}.png'} for phase in ('a','p','v')],
        {'type':'text','text':'Find lesions'}]}]}]
    source=tmp_path/'input.jsonl';source.write_text(json.dumps(rows[0])+'\n')
    settings={'seed':42,'device':'cpu','max_new_tokens':256,'image_max_pixels':262144,'image_min_pixels':1024,'do_sample':False}
    def identity():return inference_identity(model,source,tmp_path,rows,settings)
    return model,source,rows,settings,identity


@pytest.mark.parametrize('change',['seed','model','processor','image','input'])
def test_changed_inference_identity_rejected_with_same_sample_keys(tmp_path,change):
    model,source,rows,settings,identity=fixture_files(tmp_path)
    output=tmp_path/'result.csv';guard=ResumeGuard(output,identity())
    guard.partial.write_text('logical_key,slice_index\nx,0\n');guard.checkpoint(guard.partial);guard.close()
    if change=='seed':settings['seed']=2026
    elif change=='model':(model/'model.safetensors').write_bytes(b'changed weight bytes')
    elif change=='processor':(model/'preprocessor_config.json').write_text('{"resize":2}')
    elif change=='image':(tmp_path/'a.png').write_bytes(b'changed image bytes')
    else:
        rows[0]['messages'][0]['content'][-1]['text']='Changed prompt'
        source.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='changed'):ResumeGuard(output,identity())


@pytest.mark.parametrize('change',['corrupt','uncommitted','truncated','missing_sidecar'])
def test_corrupt_or_unbound_inference_output_rejected(tmp_path,change):
    *_,identity=fixture_files(tmp_path)
    output=tmp_path/'result.csv';guard=ResumeGuard(output,identity())
    guard.partial.write_text('logical_key,slice_index\nx,0\n');guard.checkpoint(guard.partial);guard.close()
    if change=='corrupt':guard.partial.write_text(guard.partial.read_text().replace('x,0','y,0'))
    elif change=='uncommitted':
        with guard.partial.open('a') as f:f.write('torn')
    elif change=='truncated':guard.partial.write_text('')
    else:guard.sidecar.unlink()
    with pytest.raises(ValueError):ResumeGuard(output,identity())


def test_clean_resume_and_completed_reuse_preserve_bytes_and_exclude_concurrent_writer(tmp_path):
    *_,identity=fixture_files(tmp_path)
    output=tmp_path/'result.csv';guard=ResumeGuard(output,identity())
    with pytest.raises(ValueError,match='Another'):ResumeGuard(output,identity())
    guard.partial.write_text('logical_key,slice_index\n');guard.checkpoint(guard.partial);guard.close()
    resumed=ResumeGuard(output,identity())
    with resumed.partial.open('a') as stream:stream.write('x,0\n')
    resumed.checkpoint(resumed.partial)
    resumed.partial.replace(output);resumed.checkpoint(output,complete=True);resumed.close()
    final=ResumeGuard(output,identity());final.checkpoint(output,complete=True);final.close()
    assert output.read_text()=='logical_key,slice_index\nx,0\n'


def test_actual_inference_entry_rejects_changed_seed_before_model_load(tmp_path,monkeypatch):
    from inference import infer_qwen_real961 as entry
    model,source,rows,settings,identity=fixture_files(tmp_path)
    output=tmp_path/'result.csv';guard=ResumeGuard(output,identity())
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=entry.FIELDNAMES);writer.writeheader()
        writer.writerow({'logical_key':'x','slice_index':0})
    guard.checkpoint(output,complete=True);guard.close()
    monkeypatch.setattr(entry.AutoProcessor,'from_pretrained',lambda *a,**k:pytest.fail('model must not load'))
    monkeypatch.setattr(sys,'argv',['infer','--model-dir',str(model),'--json-path',str(source),
        '--data-root',str(tmp_path),'--output',str(output),'--device','cpu','--seed','2026'])
    with pytest.raises(ValueError,match='changed'):entry.main()


@pytest.mark.parametrize("centers", [
    {"center_id": "source-center"},
    {"analysis_center": "source-center"},
    {"center_id": "source-center", "analysis_center": "source-center"},
])
def test_actual_inference_entry_preserves_converter_center_aliases(tmp_path, monkeypatch, centers):
    """Run the entry through CSV finalization, with only model/image decoding replaced."""
    import types
    import torch
    from inference import infer_qwen_real961 as entry
    model, source, rows, settings, identity = fixture_files(tmp_path)
    rows[0]["metadata"].update(patient_id="p", statistical_patient_id="p", **centers,
                               **{"class": "HCC", "positive": 1})
    source.write_text(json.dumps(rows[0]) + "\n")
    output = tmp_path / "predictions.csv"

    class Processor:
        tokenizer = types.SimpleNamespace(pad_token_id=None)
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, **kwargs): return {"input_ids": torch.tensor([[0]])}
        def batch_decode(self, *args, **kwargs): return ['{"detections": []}']

    class Model:
        def to(self, device): return self
        def eval(self): pass
        def generate(self, **kwargs):
            return types.SimpleNamespace(sequences=torch.tensor([[0, 1]]),
                                         logits=(torch.tensor([[0., 1.]]),),
                                         scores=(torch.tensor([[0., 1.]]),))

    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(entry.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: Model())
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", types.SimpleNamespace(
        process_vision_info=lambda messages: ([object()] * 3, None)))
    monkeypatch.setattr(sys, "argv", ["infer", "--model-dir", str(model), "--json-path", str(source),
        "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu"])
    entry.main()
    with output.open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert len(result) == 1 and result[0]["analysis_center"] == "source-center"
    assert json.loads(output.with_suffix(".csv.resume.json").read_text())["complete"] is True


def test_conflicting_inference_centers_rejected_before_model_load(tmp_path, monkeypatch):
    from inference import infer_qwen_real961 as entry
    model, source, rows, settings, identity = fixture_files(tmp_path)
    rows[0]["metadata"].update(center_id="source", analysis_center="target")
    source.write_text(json.dumps(rows[0]) + "\n")
    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: pytest.fail("model must not load"))
    monkeypatch.setattr(sys, "argv", ["infer", "--model-dir", str(model), "--json-path", str(source),
        "--data-root", str(tmp_path), "--output", str(tmp_path / "predictions.csv"), "--device", "cpu"])
    with pytest.raises(ValueError, match="center"):
        entry.main()


@pytest.mark.parametrize("module_name", ["infer_hf_vlm_revision", "infer_qwen_real961_3d_batched"])
def test_other_inference_entries_reject_unbound_old_outputs(tmp_path, monkeypatch, module_name):
    import importlib
    entry = pytest.importorskip("inference." + module_name)
    model, source, rows, settings, identity = fixture_files(tmp_path)
    output = tmp_path / "old.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=entry.FIELDNAMES)
        writer.writeheader()
        writer.writerow({"logical_key": "x", "slice_index": 0})
    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: pytest.fail("model must not load"))
    monkeypatch.setattr(sys, "argv", ["infer", "--model-dir", str(model), "--json-path", str(source),
        "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu"])
    with pytest.raises(ValueError, match="identity sidecar"):
        entry.main()


def test_batched_qwen_scores_each_sample_with_its_own_logits(tmp_path, monkeypatch):
    import types
    import torch
    entry = pytest.importorskip("inference.infer_qwen_real961_3d_batched")
    model, source, rows, settings, identity = fixture_files(tmp_path)
    rows[0]["metadata"].update(patient_id="p", statistical_patient_id="p", analysis_center="A",
                               **{"class": "HCC", "positive": 1})
    second = copy.deepcopy(rows[0]); second["metadata"]["slice_index"] = 1
    rows.append(second)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "predictions.csv"
    class Processor:
        tokenizer = types.SimpleNamespace(pad_token_id=None)
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, **kwargs): return {"input_ids": torch.tensor([[0], [0]])}
        def batch_decode(self, *args, **kwargs): return ['{"detections": []}']
    class Model:
        def to(self, device): return self
        def eval(self): pass
        def generate(self, **kwargs):
            return types.SimpleNamespace(sequences=torch.tensor([[0, 1], [0, 0]]),
                                         logits=(torch.tensor([[0., 4.], [4., 0.]]),),
                                         scores=(torch.tensor([[0., 4.], [4., 0.]]),))
    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(entry.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: Model())
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", types.SimpleNamespace(
        process_vision_info=lambda messages: ([object()] * 3, None)))
    monkeypatch.setattr(sys, "argv", ["infer", "--model-dir", str(model), "--json-path", str(source),
        "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu", "--batch-size", "2"])
    entry.main()
    with output.open(newline="") as stream:
        results = list(csv.DictReader(stream))
    expected = float(-torch.log_softmax(torch.tensor([0., 4.]), dim=-1)[1])
    assert [float(row["avg_nll"]) for row in results] == pytest.approx([expected, expected])


def test_hf_entry_bound_completion_reuse_and_model_change(tmp_path, monkeypatch):
    import types
    import torch
    from PIL import Image
    from inference import infer_hf_vlm_revision as entry
    model, source, rows, settings, identity = fixture_files(tmp_path)
    for phase in ("a", "p", "v"):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{phase}.png")
    rows[0]["metadata"].update(patient_id="p", center_id="source", **{"class": "HCC", "positive": 1})
    source.write_text(json.dumps(rows[0]) + "\n")
    output = tmp_path / "hf.csv"
    class Processor:
        tokenizer = types.SimpleNamespace(pad_token_id=None)
        def batch_decode(self, *args, **kwargs): return ['{"detections": []}']
    class Model:
        def generate(self, **kwargs):
            return types.SimpleNamespace(sequences=torch.tensor([[0, 1]]),
                                         logits=(torch.tensor([[0., 1.]]),),
                                         scores=(torch.tensor([[0., 1.]]),))
    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: Processor())
    monkeypatch.setattr(entry, "load_model", lambda *a, **k: (Model(), "synthetic"))
    monkeypatch.setattr(entry, "prepare_inputs", lambda *a, **k: {"input_ids": torch.tensor([[0]])})
    monkeypatch.setattr(sys, "argv", ["infer", "--model-dir", str(model), "--json-path", str(source),
        "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu"])
    entry.main()
    with output.open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert result[0]["analysis_center"] == "source"
    before = output.read_bytes()
    monkeypatch.setattr(entry.AutoProcessor, "from_pretrained", lambda *a, **k: pytest.fail("model must not load"))
    entry.main()
    assert output.read_bytes() == before
    (model / "model.safetensors").write_bytes(b"changed synthetic weights")
    with pytest.raises(ValueError, match="changed"):
        entry.main()
