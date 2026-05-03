import logging
import math
import random
from os.path import exists, join
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from src.utils.basic_utils import l2_normalize_np_array, load_jsonl
from src.utils.span_utils import span_xx_to_cxw
from src.utils.tensor_utils import pad_sequences_1d
from torch.utils.data import Dataset
from tqdm import tqdm
from src.vocab import Vocab

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(
        self,
        data_path: str,
        a_feat_dir: str,
        q_feat_dir: str,
        q_feat_type: str = "last_hidden_state",
        a_feat_type: str = "pann",
        max_q_l: int = 32,
        max_a_l: int = 75,
        ctx_mode: str = "video",
        clip_len: int = 2,
        max_windows: int = 5,
        span_loss_type: str = "l1",
        load_labels: bool = True,
    ) -> None:
        """Initializes the StartEndDataset.

        Args:
            data_path (str): Path to the data file.
            a_feat_dir (str): Directory for audio features.
            q_feat_dir (str): Directory for query features.
            q_feat_type (str): Type of query features.
            a_feat_type (str): Type of audio features.
            max_q_l (int): Maximum query length.
            max_a_l (int): Maximum audio length.
            ctx_mode (str): Context mode.
            clip_len (int): Clip length.
            max_windows (int): Maximum number of windows.
            span_loss_type (str): Type of span loss.
            load_labels (bool): Whether to load labels.
        """
        self.data_path = data_path
        self.a_feat_dir = a_feat_dir
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.a_feat_type = a_feat_type

        if max_a_l == -1:
            max_a_l = 100000000

        if max_q_l == -1:
            max_q_l = 100

        self.max_q_l = max_q_l
        self.max_a_l = max_a_l

        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_audio = "audio" in ctx_mode
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.load_labels = load_labels
        self.data = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        datalist = load_jsonl(self.data_path)
        return datalist

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Gets the item at the given index.

        Args:
            index (int): Index.

        Returns:
            Dict[str, Any]: Item data.
        """
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(
            meta["qid"]
        )  # (Dq, ) or (Lq, Dq)
        model_inputs["audio_feat"] = self._get_audio_feat_by_vid(meta["vid"])
        ctx_l = len(model_inputs["audio_feat"])

        # if self.use_tef:
        #     tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        #     tef_ed = tef_st + 1.0 / ctx_l
        #     tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        #     model_inputs["audio_feat"] = torch.cat(
        #         [model_inputs["audio_feat"], tef], dim=1
        #     )

        if self.use_tef:
            duration = meta["duration"]  # Total video duration in seconds
            clip_indices = torch.arange(0, ctx_l, 1.0)  # [0, 1, 2, ..., ctx_l-1]
            tef_st = (clip_indices * self.clip_len) / duration  # Normalized start times
            tef_ed = (
                (clip_indices + 1) * self.clip_len
            ) / duration  # Normalized end times
            tef_ed = torch.clamp(tef_ed, max=1.0)  # Ensure it doesn't exceed 1.0
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (ctx_l, 2)
            model_inputs["audio_feat"] = torch.cat(
                [model_inputs["audio_feat"], tef], dim=1
            )
        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(
                meta["relevant_windows"], ctx_l, meta["duration"]
            )
            (
                model_inputs["saliency_pos_labels"],
                model_inputs["saliency_neg_labels"],
                model_inputs["saliency_all_labels"],
            ) = self.get_saliency_labels_sub_as_query(
                meta["relevant_windows"][0], ctx_l
            )

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(
        self, gt_window: List[float], ctx_l: int, max_n: int = 2
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """
        Generate saliency labels for contrastive learning based on ground truth temporal window.

        This method samples positive and negative clip indices for training the saliency prediction
        head. Positive clips are sampled from within the ground truth window, while negative clips
        are sampled from outside it. Also creates a binary score array indicating relevant regions.

        Args:
            gt_window (List[float]): Ground truth time window [start_time, end_time] in seconds.
            ctx_l (int): Total number of clips/segments in the audio/video context.
            max_n (int, optional): Maximum number of positive and negative clips to sample. Defaults to 2.

        Returns:
            Tuple[List[int], List[int], np.ndarray]:
                - pos_clip_indices: List of positive clip indices (from within gt_window).
                - neg_clip_indices: List of negative clip indices (from outside gt_window).
                - score_array: Binary numpy array of shape (ctx_l,) with 1s for relevant clips.

        Example:
            >>> dataset = StartEndDataset(...)
            >>> gt_window = [26.0, 36.0]  # 10-second window
            >>> ctx_l = 75  # 75 clips total
            >>> pos_indices, neg_indices, scores = dataset.get_saliency_labels_sub_as_query(gt_window, ctx_l, max_n=2)
            >>> print(f"Positive clips: {pos_indices}")  # e.g., [13, 17]
            >>> print(f"Negative clips: {neg_indices}")  # e.g., [5, 45]
            >>> print(f"Score array shape: {scores.shape}")  # (75,)
            >>> print(f"Relevant clips: {np.where(scores == 1)[0]}")  # e.g., [13 14 15 16 17]
        """
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)

        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(
            range(gt_ed + 1, ctx_l)
        )  # to fix bugs / works..?
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        score_array = np.zeros(ctx_l)
        score_array[gt_st : gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels(
        self,
        rel_clip_ids: List[int],
        scores: List[List[float]],
        ctx_l: int,
        max_n: int = 1,
        add_easy_negative: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [
            min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]
        ]
        hard_neg_clip_indices = [
            min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]
        ]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(
        self, windows: List[List[float]], ctx_l: int, duration: float
    ) -> torch.Tensor:
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[: self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / duration  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor(
                [
                    [
                        int(w[0] / self.clip_len),
                        min(int(w[1] / self.clip_len), ctx_l) - 1,
                    ]
                    for w in windows
                ]
            ).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid: int) -> np.ndarray:
        """Gets query features by qid.

        Args:
            qid (int): Query id.

        Returns:
            np.ndarray: Query features.
        """
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat = np.load(q_feat_path)["last_hidden_state"]
        return q_feat

    def _get_audio_feat_by_vid(self, vid: str) -> torch.Tensor:
        """Gets audio features by vid.

        Args:
            vid (str): Video id.

        Returns:
            torch.Tensor: Audio features.
        """
        _feat_path = join(self.a_feat_dir, f"{vid}.npz")
        _feat = np.load(_feat_path)["features"][: self.max_a_l].astype(np.float32)
        _feat = l2_normalize_np_array(_feat)
        return torch.from_numpy(_feat)


def start_end_collate(
    batch: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collates the batch for start end dataset.

    Args:
        batch (List[Dict[str, Any]]): Batch.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: Batch meta, batched data.
    """
    batch_meta = [e["meta"] for e in batch]

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [
                dict(spans=e["model_inputs"]["span_labels"]) for e in batch
            ]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch],
                dtype=np.float32,
                fixed_length=None,
            )
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        if batch[0]["model_inputs"][k].dtype == torch.float32:
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch],
                dtype=torch.float32,
                fixed_length=None,
            )
        else:
            batched_data[k] = pad_sequences_1d(
                [torch.from_numpy(e["model_inputs"][k]) for e in batch],
                dtype=torch.float32,
                fixed_length=None,
            )
    return batch_meta, batched_data


def prepare_batch_inputs(
    batched_model_inputs: Dict[str, Any],
    device: torch.device,
    non_blocking: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """Prepares batch inputs.

    Args:
        batched_model_inputs (Dict[str, Any]): Batched model inputs.
        device (torch.device): Device.
        non_blocking (bool): Non blocking.

    Returns:
        Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]: Model inputs, targets.
    """
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(
            device, non_blocking=non_blocking
        ),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(
            device, non_blocking=non_blocking
        ),
    )

    if "audio_feat" in batched_model_inputs:
        model_inputs["src_aud"] = batched_model_inputs["audio_feat"][0].to(
            device, non_blocking=non_blocking
        )
        model_inputs["src_aud_mask"] = batched_model_inputs["audio_feat"][1].to(
            device, non_blocking=non_blocking
        )

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(
                device, non_blocking=non_blocking
            )

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(
            device, non_blocking=non_blocking
        )

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
