"""Manifest rows in VeRL's raw_prompt multimodal dataset contract."""
from __future__ import annotations

import copy
import torch
from torch.utils.data import Dataset
from rewards.schema import PHASES


class ManifestDataset(Dataset):
    def __init__(self, records, processor_kwargs):
        self.records = records
        self.processor_kwargs = processor_kwargs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        content = [{"type": "image", "image": record["images"][phase], **self.processor_kwargs}
                   for phase in PHASES]
        content.append({"type": "text", "text": record["prompt"]})
        return dict(
            raw_prompt=[{"role": "user", "content": content}], dummy_tensor=torch.tensor([0], dtype=torch.uint8),
            data_source="rl_lhm", reward_model={"style": "rule", "ground_truth": record["ground_truth"]},
            extra_info={"index": index, "sample_uid": record["uid"]}, index=index,
            tools_kwargs={}, interaction_kwargs={}, agent_name="rl_lhm_single_turn",
        )


def occurrence_rows(dataset, indices, *, epoch, offset, batch_size, group_size, seeds):
    """Pad a short global prompt batch; retain distinct RID and persistent UID.

    All ranks receive equal work; padded rows retain attention tokens but have
    zero loss masks and never enter reward/baseline/curriculum populations.
    """
    if not indices or len(indices) > batch_size or len(seeds) != batch_size * group_size:
        raise ValueError("invalid global prompt batch or per-completion random seeds")
    rows, records, rids, real = [], [], [], []
    for slot in range(batch_size):
        keep = slot < len(indices)
        index = indices[slot] if keep else indices[0]
        rid = f"{epoch}:{offset + slot}" if keep else f"padding:{epoch}:{offset + slot}"
        for member in range(group_size):
            row = copy.deepcopy(dataset[index])
            row["uid"] = rid
            row["extra_info"].update(rl_lhm_seed=seeds[len(rows)], rid=rid, member=member, real=keep)
            # Stable, occurrence-based trace/request identity, never the persistent UID.
            row["index"] = epoch * len(dataset) + offset + slot
            rows.append(row)
            records.append(dataset.records[index])
            rids.append(rid)
            real.append(keep)
    return rows, records, rids, real
