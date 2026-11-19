# Copyright (c) FunEmbed Contributors.
"""Describe-then-Embed training plugin for ms-swift.

Load via `--custom_register_path /path/to/describe_then_embed_plugin.py` together with
`--loss_type describe_then_embed`, `--template describe_then_embed` and
`--new_special_tokens '<gen_emb>'`.

This plugin ONLY adds new names into the existing registries (loss_map + template mapping);
it does not modify any core ms-swift source file.
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

from swift.loss import BaseLoss, loss_map
from swift.loss.embedding import InfonceLoss
from swift.metrics import eval_metrics_map
from swift.metrics.base import EvalMetrics
from swift.model.patcher import get_lm_head_model
from swift.model.utils import get_llm_model
from swift.template.register import register_template
from swift.template.templates.qwen import Qwen3VLEmbTemplate, QwenTemplateMeta
from swift.utils import get_logger

logger = get_logger()

# Running loss instance (set in DescribeThenEmbedLoss.__init__); lets the logging
# callback read per-term loss components. Per-rank in DDP (each process has its own).
_LOSS_INSTANCE = None


class DescribeThenEmbedMetrics(EvalMetrics):
    """Retrieval metric for the describe-then-embed layout.

    The collator emits, per sample, exactly 18 embeddings in order
    `[anchor, positive, negative_0 .. negative_15]`. We group the eval
    predictions into chunks of 18 and measure how often the positive ranks
    first (retrieval top-1) plus the mean reciprocal rank of the positive.
    """

    def __init__(self, args, trainer):
        super().__init__(args, trainer)

    def compute_metrics(self, eval_prediction):
        preds = eval_prediction.predictions
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        preds = preds.float()
        if preds.ndim != 2 or preds.shape[0] < 18:
            return {}
        group = 18
        n_groups = preds.shape[0] // group
        preds = preds[:n_groups * group].reshape(n_groups, group, -1)
        anchor = F.normalize(preds[:, 0], dim=-1)   # [G, D]
        docs = F.normalize(preds[:, 1:], dim=-1)    # [G, 17, D]
        sims = torch.einsum('gd,gkd->gk', anchor, docs)  # [G, 17]
        # `positive` is docs[:, 0]
        top1 = (sims.argmax(dim=1) == 0).float().mean().item()
        rank = (sims > sims[:, 0:1]).sum(dim=1).float()
        mrr = (1.0 / (rank + 1.0)).mean().item()
        return {'retrieval_top1': top1, 'retrieval_mrr': mrr}


eval_metrics_map['describe_then_embed'] = DescribeThenEmbedMetrics

GEN_EMB_TOKEN = '<gen_emb>'
# number of consecutive <gen_emb> tokens appended per sequence (>=1);
# the representation vector is the L2-normalised mean of the hidden states
# at all <gen_emb> positions (mean pooling).
GEN_EMB_NUM = max(1, int(os.environ.get('GEN_EMB_NUM', '1')))


def _init_gen_emb_from_eos(model, tokenizer, gen_emb_id):
    """Copy the eos token's embedding row into the <gen_emb> row (one-shot, in place).

    Called right before the first forward. The base model's embed_tokens is frozen
    in this training setup, so <gen_emb>'s input vector would otherwise stay at its
    random init; initialising it from <eos> gives the pooled positions a sane
    starting point. Safe to call multiple times (idempotent).
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None or eos_id == gen_emb_id:
        return False
    with torch.no_grad():
        embed = model.get_input_embeddings()
        weight = embed.weight
        if weight.shape[0] <= max(eos_id, gen_emb_id):
            return False
        src, dst = weight[eos_id], weight[gen_emb_id]
        if not torch.allclose(src, dst):
            weight[gen_emb_id].copy_(src)
            return True
    return False


class DescribeThenEmbedLoss(BaseLoss):
    """Joint loss L = lambda_gen * L_gen + L_contrastive.

    - L_contrastive: InfoNCE over the `<gen_emb>` embeddings (reuses ms-swift InfonceLoss,
      non-batch mode with fixed K negatives).
    - L_gen: token-level cross entropy over the description tokens (Query caption + Doc
      dictionary description), computed by re-applying the original lm_head weights onto the
      full last hidden states H.
    - DESC_EMBED_LAMBDA weights the GENERATIVE term: generative learning needs a larger
      effective learning rate than contrastive learning, so L_gen is scaled by lambda_gen.
    """

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.lambda_gen = float(os.environ.get('DESC_EMBED_LAMBDA', '5.0'))
        # Fraction of the total training steps during which the generative loss L_gen
        # participates. GEN_LOSS_RATIO=0.2 keeps L_gen only for the first 20% of steps;
        # afterwards the loss is contrastive-only. 1.0 (default) keeps L_gen always on.
        self.gen_loss_ratio = float(os.environ.get('GEN_LOSS_RATIO', '1.0'))
        self.gen_emb_num = GEN_EMB_NUM
        self._gen_emb_init_done = False
        self._infonce = InfonceLoss(args, trainer)
        self._lm_head = None
        self._setup_done = False
        logger.info(
            f'[DescribeThenEmbed] lambda_gen={self.lambda_gen} '
            f'gen_loss_ratio={self.gen_loss_ratio} gen_emb_num={self.gen_emb_num}')

        global _LOSS_INSTANCE
        _LOSS_INSTANCE = self
        # 分项损失走 trainer.custom_metrics:ms-swift 的 Trainer.log() 会自动
        # compute+reset 并写入 logging.jsonl(transformers on_log 钩子会被 mixin.log 绕过)。

    def _report(self, name, value):
        """累计一个分项到 trainer 的 custom_metrics['train']（log 时自动求均值）。"""
        metrics = getattr(self.trainer, 'custom_metrics', None)
        if metrics is None:
            return
        metrics['train'][name].update(value)

    def _gen_loss_active(self) -> bool:
        if self.gen_loss_ratio >= 1.0:
            return True
        state = getattr(self.trainer, 'state', None)
        if state is None:
            return True
        max_steps = getattr(state, 'max_steps', None)
        if not max_steps:
            return True
        step = getattr(state, 'global_step', 0) or 0
        return (step / max_steps) < self.gen_loss_ratio

    def _setup(self):
        if self._setup_done:
            return
        self._setup_done = True
        model = self.trainer.model
        model_meta = self.trainer.model_meta
        lm_head_model = get_lm_head_model(model, model_meta=model_meta)
        lm_head = getattr(lm_head_model, 'lm_head', None)
        if not isinstance(lm_head, nn.Linear):
            raise RuntimeError(
                f'[DescribeThenEmbed] cannot locate a nn.Linear `lm_head`; found {type(lm_head)!r}.')
        self._lm_head = lm_head
        if not self._gen_emb_init_done:
            copied = _init_gen_emb_from_eos(
                self.trainer.model, self.trainer.template.tokenizer,
                self.trainer.template.gen_emb_token_id)
            self._gen_emb_init_done = True
            if copied:
                logger.info('[DescribeThenEmbed] <gen_emb> input embedding initialised '
                            'from <eos> embedding row.')

    def _pooled_embedding(self, E_last, H, template):
        """Mean-pool the hidden states at all <gen_emb> positions, then L2-normalise.

        Falls back to the framework-provided last-token embedding (E_last) when the
        position mask is unavailable, so GEN_EMB_NUM=1 without a mask behaves exactly
        like the original single-token protocol.
        """
        mask = getattr(template, '_last_gen_emb_mask', None)
        if mask is None or H is None or not bool(mask.any()):
            return E_last
        pooled = (H * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        return pooled

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        self._setup()
        template = self.trainer.template
        E_last = outputs['last_hidden_state']  # [N, D] framework last-valid-token embedding
        H = template._last_H  # [N, T, D]
        gen_labels = template._last_gen_labels  # [N, T] int64
        E = self._pooled_embedding(E_last, H, template)

        loss_contrastive = self._infonce({'last_hidden_state': E}, labels)

        loss_gen = E.new_zeros(())
        if self._gen_loss_active():
            if H is None:
                raise RuntimeError('[DescribeThenEmbed] `_last_H` is None; the H-capture hook did not fire.')
            if gen_labels is not None:
                rolled = torch.roll(gen_labels, shifts=-1, dims=-1)
                mask = rolled != -100
                if mask.any():
                    H_sup = H[mask]  # [S, D]
                    targets = rolled[mask].long()
                    Z = F.linear(H_sup, self._lm_head.weight, self._lm_head.bias)
                    loss_gen = F.cross_entropy(Z.float(), targets)

        self._report('loss_gen', loss_gen.detach().item())
        self._report('loss_contrastive', loss_contrastive.detach().item())
        return self.lambda_gen * loss_gen + loss_contrastive


loss_map['describe_then_embed'] = DescribeThenEmbedLoss


class DescribeThenEmbedTemplate(Qwen3VLEmbTemplate):
    gen_emb_token = GEN_EMB_TOKEN

    def _init_placeholder_tokens(self):
        super()._init_placeholder_tokens()
        self.placeholder_tokens = list(self.placeholder_tokens)
        # Doc-side generation supervision scope:
        #   all      - supervise positive + all negative doc descriptions (default)
        #   positive - supervise only the positive doc description (negatives masked)
        #   none     - supervise no doc description (anchor caption only)
        # Backwards-compatible aliases: true -> all, false -> none.
        doc_gen = os.environ.get('DOC_GEN_SUPERVISE', 'all').strip().lower()
        doc_gen = {'true': 'all', 'false': 'none'}.get(doc_gen, doc_gen)
        if doc_gen not in {'all', 'positive', 'none'}:
            raise ValueError(
                f"[DescribeThenEmbed] DOC_GEN_SUPERVISE={doc_gen!r} is not one of "
                "{'all', 'positive', 'none'}.")
        self.doc_gen_supervise = doc_gen
        attn_mode = os.environ.get('GEN_EMB_ATTN', 'default')
        valid_attn_modes = {'default', 'caption_only', 'image_only'}
        if attn_mode not in valid_attn_modes:
            raise ValueError(
                f'[DescribeThenEmbed] GEN_EMB_ATTN={attn_mode!r} is not one of {sorted(valid_attn_modes)}.')
        self.gen_emb_attn_mode = attn_mode
        gen_emb_id = self.tokenizer.convert_tokens_to_ids(self.gen_emb_token)
        if gen_emb_id is None:
            raise ValueError(
                f"[DescribeThenEmbed] token {self.gen_emb_token!r} not found in the tokenizer. "
                f"Please pass `--new_special_tokens '<gen_emb>'`.")
        self.gen_emb_token_id = gen_emb_id
        if self.gen_emb_token_id not in self.placeholder_tokens:
            self.placeholder_tokens.append(self.gen_emb_token_id)
        self.gen_emb_num = GEN_EMB_NUM

    def _encode(self, inputs):
        encoded = super()._encode(inputs)
        gen_emb_id = self.gen_emb_token_id
        n = self.gen_emb_num
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        gen_labels = list(labels) if labels is not None else [-100] * len(input_ids)
        # <gen_emb> positions marker (for mean pooling over the pooled tokens)
        gen_emb_mask = [0] * len(input_ids) + [1] * n
        input_ids = list(input_ids) + [gen_emb_id] * n
        if labels is not None:
            labels = list(labels) + [-100] * n
        gen_labels = list(gen_labels) + [-100] * n
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['gen_labels'] = gen_labels
        encoded['gen_emb_mask'] = gen_emb_mask
        loss_scale = encoded.get('loss_scale')
        if loss_scale is not None:
            encoded['loss_scale'] = list(loss_scale) + [0] * n
        mm_token_type_ids = encoded.get('mm_token_type_ids')
        if mm_token_type_ids is not None:
            if not isinstance(mm_token_type_ids, torch.Tensor):
                mm_token_type_ids = torch.tensor(mm_token_type_ids)
            encoded['mm_token_type_ids'] = torch.cat(
                [mm_token_type_ids, torch.zeros(n, dtype=mm_token_type_ids.dtype)])
        return encoded

    def _truncate(self, input_ids, labels, encoded, truncation_strategy):
        gen_labels = encoded.get('gen_labels')
        if gen_labels is not None:
            placeholder_tokens = torch.tensor(self.placeholder_tokens)
            input_ids_tensor = torch.tensor(input_ids)
            protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
            n_protected = protected.sum().item()
            if n_protected < self.max_length:
                non_protected = (~protected).nonzero(as_tuple=True)[0]
                if truncation_strategy == 'left':
                    idx = non_protected[-(self.max_length - n_protected):]
                else:
                    idx = non_protected[:self.max_length - n_protected]
                protected[idx] = True
            gen_labels = torch.tensor(gen_labels)[protected].tolist()
            gen_labels[0] = -100
            encoded['gen_labels'] = gen_labels
        gen_emb_mask = encoded.get('gen_emb_mask')
        if gen_emb_mask is not None:
            # keep the mask aligned with the truncated input_ids
            keep = torch.tensor(gen_emb_mask[:len(input_ids)])
            if len(keep) < len(input_ids):
                keep = torch.cat([keep, torch.zeros(len(input_ids) - len(keep), dtype=keep.dtype)])
            encoded['gen_emb_mask'] = keep.tolist()
        return super()._truncate(input_ids, labels, encoded, truncation_strategy)

    def _post_encode(self, model, inputs):
        self._last_gen_labels = inputs.pop('gen_labels', None)
        self._last_gen_emb_mask = inputs.pop('gen_emb_mask', None)
        self._ensure_H_capture_hook(model)
        return super()._post_encode(model, inputs)

    def _ensure_H_capture_hook(self, model):
        if getattr(self, '_H_hook_registered', False):
            return
        llm_model = get_llm_model(model, model_meta=self.model_meta)

        def _capture_H(module, args, kwargs, output):
            if hasattr(output, 'last_hidden_state'):
                self._last_H = output.last_hidden_state
            else:
                self._last_H = output[0]

        llm_model.register_forward_hook(_capture_H, with_kwargs=True)
        self._H_hook_registered = True
        logger.info(f'[DescribeThenEmbed] registered H-capture hook on {llm_model.__class__.__name__}.')

        if self.gen_emb_attn_mode != 'default':
            def _apply_ablation_mask(module, args, kwargs):
                inputs_embeds = kwargs.get('inputs_embeds')
                if inputs_embeds is not None and 'attention_mask' in kwargs:
                    kwargs = dict(kwargs)
                    kwargs['attention_mask'] = self._build_ablation_4d_mask(
                        inputs_embeds,
                        kwargs['attention_mask'],
                        kwargs.get('visual_pos_masks'),
                        getattr(self, '_last_gen_labels', None),
                    )
                return args, kwargs

            llm_model.register_forward_pre_hook(_apply_ablation_mask, with_kwargs=True)
            logger.info(
                f'[DescribeThenEmbed] registered GEN_EMB_ATTN ablation-mask pre-hook '
                f'(mode={self.gen_emb_attn_mode}) on {llm_model.__class__.__name__}.')

    def _build_ablation_4d_mask(self, inputs_embeds, attention_mask, visual_pos_masks, gen_labels):
        r"""Build a custom 4D additive mask restricting the `<gen_emb>` query row.

        All rows keep the standard causal + padding pattern. For the `<gen_emb>` row (the
        rightmost valid token of each sequence) the attended key/value columns are restricted to
        `caption_only` (token-level description labels) or `image_only` (visual token positions).
        """
        B, T = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        if attention_mask is not None:
            valid = attention_mask.bool().to(device)
        else:
            valid = torch.ones(B, T, dtype=torch.bool, device=device)

        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        mask = causal[None, None, :, :] & valid[:, None, None, :]  # [B, 1, T, T]

        if self.gen_emb_attn_mode == 'caption_only':
            allowed = gen_labels != -100 if gen_labels is not None else torch.zeros(
                B, T, dtype=torch.bool, device=device)
        elif self.gen_emb_attn_mode == 'image_only':
            allowed = visual_pos_masks.bool().to(device) if visual_pos_masks is not None else torch.zeros(
                B, T, dtype=torch.bool, device=device)
        else:  # pragma: no cover - guarded by _init_placeholder_tokens
            raise ValueError(f'[DescribeThenEmbed] unknown GEN_EMB_ATTN mode: {self.gen_emb_attn_mode!r}')

        gen_emb_pos = valid.sum(dim=1) - 1  # [B]; index of the last valid token (<gen_emb>)
        diag = torch.arange(T, device=device)[None, :] == gen_emb_pos[:, None]  # keep self-attention
        allowed = (allowed | diag).to(device)

        for b in range(B):
            gp = gen_emb_pos[b]
            mask[b, 0, gp] = causal[gp] & valid[b] & allowed[b]

        return mask

    def _embedding_data_collator(self, batch, *, padding_to=None):
        labels = []
        new_batch = []
        gen_labels_rows = []
        gen_emb_mask_rows = []
        doc_gen_supervise = getattr(self, 'doc_gen_supervise', 'all')
        for b in batch:
            if 'input_ids' in b:
                new_batch += [b]
                gen_labels_rows += [b.get('gen_labels')]
                gen_emb_mask_rows += [b.get('gen_emb_mask')]
            else:
                keys = [key for key in b.keys() if 'negative' in key]
                max_neg = None
                for key in keys:
                    value_list = b[key]
                    suffix = key[len('negative_'):]
                    max_neg = len(value_list)
                    for i, value in enumerate(value_list):
                        b[f'negative{i}_{suffix}'] = value
                    b.pop(key)

                indexes = ['anchor_', 'positive_']
                if max_neg is not None:
                    for i in range(0, max_neg):
                        indexes.append(f'negative{i}_')
                for prefix in indexes:
                    new_batch += self._fetch_inputs_startswith([b], prefix)
                    gl = b.get(prefix + 'gen_labels')
                    gm = b.get(prefix + 'gen_emb_mask')
                    if gl is not None and prefix != 'anchor_':
                        mask = doc_gen_supervise == 'none' or (doc_gen_supervise == 'positive'
                                                               and prefix.startswith('negative'))
                        if mask:
                            gl = [-100] * len(gl)
                    gen_labels_rows += [gl]
                    gen_emb_mask_rows += [gm if gm is not None else []]
            labels.extend(b.get('labels', []))
        res = self._data_collator(new_batch, padding_to=padding_to)
        if labels:
            res['labels'] = torch.tensor(labels, dtype=torch.float32)
        if any(gl is not None for gl in gen_labels_rows):
            if 'input_ids' in res:
                max_len = res['input_ids'].shape[-1]
            else:
                max_len = max(len(gl) for gl in gen_labels_rows if gl is not None)
            padded = []
            for gl in gen_labels_rows:
                if gl is None:
                    gl = []
                padded.append(list(gl) + [-100] * (max_len - len(gl)))
            res['gen_labels'] = torch.tensor(padded, dtype=torch.long)
        if any(len(gm) > 0 for gm in gen_emb_mask_rows):
            max_len = res['input_ids'].shape[-1]
            padded = []
            for gm in gen_emb_mask_rows:
                gm = gm or []
                padded.append(list(gm) + [0] * (max_len - len(gm)))
            res['gen_emb_mask'] = torch.tensor(padded, dtype=torch.bool)
        return res


register_template(
    QwenTemplateMeta(
        'describe_then_embed',
        default_system="Represent the user's input.",
        suffix=['<|endoftext|>'],
        template_cls=DescribeThenEmbedTemplate,
    ))