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
