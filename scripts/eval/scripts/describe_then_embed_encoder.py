"""Describe-then-embed encoder plugin for the funembed-eval framework.

Faithful two-step representation for the trained describe-then-embed paradigm,
reusing the training-side ms-swift framework (``framework_path``) and the
training plugin (template + LoRA adapter) — no evaluation-side model code:

  1. GENERATE (causal_lm engine): run base+LoRA with the training-side
     query/doc instructions (v1/prompts/*_instruction_prompt.json) and greedy
     sampling (max 256 new tokens) to describe each eval branch.
  2. EMBED (embedding engine): feed the generated description text through the
     same base+LoRA and take the last-valid-input-token final hidden state,
     L2-normalised (the protocol the model was aligned on).

Contract required by funembed-eval ``encoder.backend=plugin``:
  create_encoder(config) -> object with .report: dict and
  .encode(branches) -> (np.float32 [N, d] unit-norm rows, [per-branch token count]).

Framework branches arrive as
  {'messages': [{'role': 'system', 'content': <framework instruction>},
                {'role': 'user',   'content': '<image>'*n or candidate text}],
   'images': [...]} (images only for image branches)

The framework system text is intentionally replaced by the training-side
instructions (decided with the model owner): generation quality only makes
sense in-distribution.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

GENERATION_MAX_NEW_TOKENS = 256
DEFAULT_SYSTEM = "Represent the user's input."
# Number of consecutive <gen_emb> tokens used for representation; the vector is
# the L2-normalised mean of their final hidden states (mean pooling). 1 = original
# single-token protocol. Must match the training-side GEN_EMB_NUM.
GEN_EMB_NUM = max(1, int(os.environ.get('GEN_EMB_NUM', '1')))
# 训练侧 prompt：默认取同仓训练目录 v1 的 prompts/，可用环境变量 DTE_TRAIN_DIR 覆盖
_TRAIN_V1_DIR = Path(os.environ.get('DTE_TRAIN_DIR',
                                    Path(__file__).resolve().parents[2] / 'train' / 'v1'))
_TEMPLATES_DIR = _TRAIN_V1_DIR / 'prompts'

# Fallbacks mirror v1/prompts/*.json verbatim.
_QUERY_FALLBACK = {
    'C': 'CFP Image: <image>\nDescribe the visual features of the Color Fundus Photography (CFP) image.',
    'O': 'OCT Scans: <image>...<image>\nDescribe the visual features of the Optical Coherence Tomography (OCT) scans.',
    'B': 'CFP Image: <image>\nOCT Scans: <image>...<image>\n'
         'Describe the visual features of the provided images separately:',
}
_DOC_FALLBACK = {
    'C': 'Disease: {DISEASE_NAME}\nDescribe the visual features for this disease in Color Fundus Photography (CFP).',
    'O': 'Disease: {DISEASE_NAME}\nDescribe the visual features for this disease in Optical Coherence Tomography (OCT).',
    'B': 'Disease: {DISEASE_NAME}\nDescribe the visual features for this disease separately in '
         'Color Fundus Photography (CFP) and Optical Coherence Tomography (OCT).',
}


def _load_training_instructions():
    try:
        q = json.loads((_TEMPLATES_DIR / 'query_instruction_prompt.json').read_text(encoding='utf-8'))
        d = json.loads((_TEMPLATES_DIR / 'doc_instruction_prompt.json').read_text(encoding='utf-8'))
        return ({'C': q['cfp_only'], 'O': q['oct_only'], 'B': q['cfp_oct_combined']},
                {'C': d['cfp_only'], 'O': d['oct_only'], 'B': d['cfp_oct_combined']})
    except Exception:
        return dict(_QUERY_FALLBACK), dict(_DOC_FALLBACK)


def _format_label_for_prompt(raw_label):
    words = [w for w in raw_label.strip().split('_') if w]
    if not words:
        return raw_label
    return ' '.join(w[:1].upper() + w[1:] for w in words)


def _modality_from_branch(branch):
    """C/O/B for image branches by image count (protocol: C=1, O=6, B=7); None for text."""
    n_img = len(branch.get('images') or [])
    if n_img == 0:
        return None
    if n_img >= 7:
        return 'B'
    if n_img >= 2:
        return 'O'
    return 'C'


def _doc_modality(branch):
    """Target modality for a text (disease) branch, inferred from the framework system text."""
    sys_txt = ' '.join(m['content'] for m in branch['messages'] if m['role'] == 'system').lower()
    has_oct, has_cfp = 'oct' in sys_txt, 'fundus' in sys_txt
    if has_oct and has_cfp:
        return 'B'
    if has_oct:
        return 'O'
    return 'C'



def _image_query_user(mod, n_img):
    """Rebuild the training-side query user message with the exact image count."""
    if mod == 'C':
        return 'CFP Image: ' + '<image>' * n_img + \
               '\nDescribe the visual features of the Color Fundus Photography (CFP) image.'
    if mod == 'O':
        return 'OCT Scans: ' + '<image>' * n_img + \
               '\nDescribe the visual features of the Optical Coherence Tomography (OCT) scans.'
    return 'CFP Image: <image>\nOCT Scans: ' + '<image>' * (n_img - 1) + \
           '\nDescribe the visual features of the provided images separately:'


class DescribeThenEmbedEncoder:
    def __init__(self, config):
        e = config['encoder']
        framework = e.get('framework_path')
        if framework:
            sys.path.insert(0, framework)
        os.environ.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
                          IMAGE_MAX_TOKEN_NUM=str(e.get('image_max_tokens', 1024)))

        # Register the describe_then_embed template (training plugin) so the
        # training-side template/system suffix is available if needed.
        train_plugin = e.get('train_plugin') or str(_TRAIN_V1_DIR / 'scripts' / 'describe_then_embed_plugin.py')
        spec = importlib.util.spec_from_file_location('dte_train_plugin', train_plugin)
        module = importlib.util.module_from_spec(spec)
        sys.modules['dte_train_plugin'] = module
        spec.loader.exec_module(module)

        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

        def make(task_type, template_type):
            return TransformersEngine(
                e['model'],
                adapters=[e['adapter']] if e.get('adapter') else None,
                model_type='qwen3_vl_emb',
                template_type=template_type,
                task_type=task_type,
                torch_dtype=torch.bfloat16,
                attn_impl='sdpa',
                device_map=e.get('device', 'cuda:0'),
                max_batch_size=e.get('batch_size', 1),
                new_special_tokens=['<gen_emb>'],
            )

        # Generation engine: causal_lm task so infer() actually samples tokens.
        self.gen_engine = make('causal_lm', 'qwen3_vl')
        # Embedding engine: last-valid-input-token final hidden state + L2.
        self.emb_engine = make('embedding', 'qwen3_vl_emb')
        for eng in (self.gen_engine, self.emb_engine):
            eng.model.requires_grad_(False)
            eng.model.eval()

        # <gen_emb> input embedding initialised from <eos> (idempotent, matches training)
        tok = self.emb_engine.processor.tokenizer
        gen_emb_id = tok.convert_tokens_to_ids('<gen_emb>')
        if gen_emb_id is not None and tok.eos_token_id is not None and gen_emb_id != tok.eos_token_id:
            with torch.no_grad():
                w = self.emb_engine.model.get_input_embeddings().weight
                if w.shape[0] > max(gen_emb_id, tok.eos_token_id) and not torch.allclose(
                        w[gen_emb_id], w[tok.eos_token_id]):
                    w[gen_emb_id].copy_(w[tok.eos_token_id])
        # Keep both engines' embeddings in sync (separate model instances)
        if self.gen_engine is not self.emb_engine:
            with torch.no_grad():
                we = self.emb_engine.model.get_input_embeddings().weight
                wg = self.gen_engine.model.get_input_embeddings().weight
                if wg.shape == we.shape:
                    wg[gen_emb_id].copy_(we[gen_emb_id])

        self.gen_emb_num = GEN_EMB_NUM

        self.InferRequest = InferRequest
        self.torch = torch
        self.RequestConfig = RequestConfig

        self.query_instr, self.doc_instr = _load_training_instructions()
        self.classes = json.loads(Path(config['classes']).read_text(encoding='utf-8'))
        # generated-description export (off unless run_eval.sh / config enables it)
        self.descriptions_dir = e.get('descriptions_dir') if e.get('save_descriptions') else None
        self._label_by_normtext = {
            c['text'].strip().rstrip('.').lower(): _format_label_for_prompt(c['label']) for c in self.classes
        }

        self.report = {
            'backend': 'plugin',
            'implementation': 'describe_then_embed_two_step',
            'representation_protocol':
                f'generate description (training-side instructions, greedy, max_new_tokens='
                f'{GENERATION_MAX_NEW_TOKENS}) then embed generated text: mean over the final '
                f'hidden states of GEN_EMB_NUM={GEN_EMB_NUM} trailing <gen_emb> tokens (init '
                f'from <eos>), L2; doc side generates from training doc instruction',
            'model': e['model'],
            'adapter': e.get('adapter'),
            'framework_path': framework,
            'train_plugin': train_plugin,
            'gen_template': 'qwen3_vl (causal_lm)',
            'emb_template': 'qwen3_vl_emb (embedding)',
            'query_instruction': self.query_instr,
            'doc_instruction': self.doc_instr,
            'system': DEFAULT_SYSTEM,
            'max_new_tokens': GENERATION_MAX_NEW_TOKENS,
            'train_new_special_tokens': ['<gen_emb>'],
            'save_descriptions': bool(self.descriptions_dir),
            'descriptions_dir': self.descriptions_dir,
        }

    # ---------------- steps ----------------
    def _generate(self, requests):
        cfg = self.RequestConfig(temperature=0, max_tokens=GENERATION_MAX_NEW_TOKENS)
        outs = self.gen_engine.infer(requests, cfg, use_tqdm=False)
        texts = []
        for r in outs:
            if isinstance(r, Exception):
                raise r
            texts.append(r.choices[0].message.content.strip())
        return texts

    def _embed(self, texts):
        """Embed generated texts.

        GEN_EMB_NUM=1: delegate to the embedding engine's last-valid-token protocol.
        GEN_EMB_NUM>1: run a manual forward with N trailing <gen_emb> tokens and
        mean-pool their final hidden states (L2-normalised), matching training.
        """
        if self.gen_emb_num <= 1:
            reqs = [self.InferRequest(messages=[{'role': 'system', 'content': DEFAULT_SYSTEM},
                                                {'role': 'user', 'content': t}]) for t in texts]
            outs = self.emb_engine.infer(reqs, self.RequestConfig(), use_tqdm=False)
            vecs, toks = [], []
            for r in outs:
                if isinstance(r, Exception):
                    raise r
                emb = np.asarray(r.data[0].embedding, np.float64)
                n = float(np.linalg.norm(emb))
                if n == 0 or not np.isfinite(n):
                    raise ValueError('non-finite or zero embedding')
                vecs.append(emb / n)
                toks.append(int(r.usage.prompt_tokens))
            return np.asarray(vecs, np.float32), toks

        # Multi-token path: manual forward + mean pooling over <gen_emb> positions.
        torch = self.torch
        tok = self.emb_engine.processor.tokenizer
        gen_emb_id = tok.convert_tokens_to_ids('<gen_emb>')
        if gen_emb_id is None:
            raise ValueError('<gen_emb> token not registered in embedding engine tokenizer')
        engine = self.emb_engine
        engine.model.eval()
        captured = {}

        def _capture(module, args, kwargs, output):
            hidden = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output[0]
            captured['hidden'] = hidden

        from swift.model.utils import get_llm_model
        from swift.utils import to_device
        llm = get_llm_model(engine.model, model_meta=engine.model_meta)
        handle = llm.register_forward_hook(_capture, with_kwargs=True)
        try:
            vecs, toks = [], []
            for t in texts:
                req = self.InferRequest(messages=[
                    {'role': 'system', 'content': DEFAULT_SYSTEM},
                    {'role': 'user', 'content': t + ' <gen_emb>' * self.gen_emb_num}])
                encoded = engine.template.encode(req, return_template_inputs=True)
                inputs = to_device(engine.template.data_collator([encoded]), engine.model.device)
                if engine.model_meta.is_multimodal:
                    _, inputs = engine.template.pre_forward_hook(engine.model, None, inputs)
                fwd_keys = ('input_ids', 'attention_mask', 'position_ids', 'past_key_values',
                            'inputs_embeds', 'use_cache', 'output_hidden_states')
                with torch.no_grad():
                    engine.model(**{k: v for k, v in inputs.items() if k in fwd_keys})
                hidden = captured['hidden']  # [1, T, D]
                input_ids = inputs['input_ids'][0].tolist()
                positions = [i for i, tid in enumerate(input_ids) if tid == gen_emb_id]
                positions = positions[-self.gen_emb_num:]
                if not positions:
                    raise ValueError('no <gen_emb> tokens found in embedded sequence')
                pooled = hidden[0, positions, :].float().mean(dim=0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                vecs.append(pooled.cpu().numpy().astype(np.float64))
                toks.append(int(inputs['input_ids'].shape[1]))
            return np.asarray(vecs, np.float32), toks
        finally:
            handle.remove()

    # ---------------- framework contract ----------------
    def encode(self, branches):
        gen_reqs = []
        metas = []
        for b in branches:
            images = b.get('images')
            if images:
                mod = _modality_from_branch(b)
                user = _image_query_user(mod, len(images))
                kind = 'image_' + mod
                gen_reqs.append(self.InferRequest(
                    messages=[{'role': 'system', 'content': DEFAULT_SYSTEM},
                              {'role': 'user', 'content': user}],
                    images=list(images)))
            else:
                text = b['messages'][-1]['content']
                mod = _doc_modality(b)
                label = self._label_by_normtext.get(
                    text.strip().rstrip('.').lower(), _format_label_for_prompt(text))
                user = self.doc_instr[mod].replace('{DISEASE_NAME}', label)
                kind = 'doc_' + mod
                gen_reqs.append(self.InferRequest(
                    messages=[{'role': 'system', 'content': DEFAULT_SYSTEM},
                              {'role': 'user', 'content': user}]))
            metas.append({'kind': kind, 'prompt': user})

        generated = self._generate(gen_reqs)
        vecs, toks = self._embed(generated)
        self._dump_descriptions(metas, generated)

        norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
        if not np.all(np.isfinite(vecs)):
            raise ValueError('non-finite vectors')
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            raise ValueError('vectors not unit norm after fp32')
        return vecs, toks

    def _dump_descriptions(self, metas, generated):
        """Append this batch's generated descriptions to descriptions/<kind>.jsonl."""
        out_dir = self.descriptions_dir
        if not out_dir:
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        by_kind = {}
        for meta, text in zip(metas, generated):
            by_kind.setdefault(meta['kind'], []).append(
                {'prompt': meta['prompt'], 'description': text})
        for kind, items in by_kind.items():
            with open(Path(out_dir) / f'{kind}.jsonl', 'a', encoding='utf-8') as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_encoder(config):
    return DescribeThenEmbedEncoder(config)
