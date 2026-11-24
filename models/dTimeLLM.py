"""dTimeLLM: Prophet-inspired T/S/R decomposition + early-concatenation Time-LLM.

Based on the Time-LLM architecture with a frozen LLM, shared
reprogramming layer, component-specific patch embeddings and full FlattenHead.
This model forecasts the full pred_len in ONE pass, unlike TimeLLM_ISF.
"""

from math import sqrt, pi
from contextlib import nullcontext
import os

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)
        self.n_changepoints = int(getattr(configs, 'prophet_n_changepoints',
                                          getattr(configs, 'n_changepoints', 10)))
        self.changepoint_range = float(getattr(configs, 'prophet_changepoint_range',
                                               getattr(configs, 'changepoint_range', 0.8)))
        self.changepoint_prior_scale = float(getattr(configs, 'prophet_changepoint_prior_scale',
                                                     getattr(configs, 'changepoint_prior_scale', 0.05)))
        self.fourier_order = int(getattr(configs, 'prophet_fourier_order',
                                         getattr(configs, 'fourier_order', 5)))
        self.seasonality_prior_scale = float(getattr(configs, 'prophet_seasonality_prior_scale',
                                                     getattr(configs, 'seasonality_prior_scale', 10.0)))
        # Optional environment override avoids requiring new argparse flags.
        raw_period = os.environ.get('DTIMELLM_PERIOD') or getattr(configs, 'prophet_seasonal_periods', None)
        if raw_period is None:
            raw_period = getattr(configs, 'period', 24)
        if isinstance(raw_period, str):
            raw_period = raw_period.split(',')[0].strip()
        elif isinstance(raw_period, (list, tuple)):
            raw_period = raw_period[0]
        self.seasonal_period = float(raw_period)
        if self.seasonal_period <= 1:
            raise ValueError('seasonal_period must be > 1')
        if self.patch_len <= 0 or self.stride <= 0:
            raise ValueError('Patch length and stride must be > 0')

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = False
            self.llama_config.output_hidden_states = False
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = False
            self.gpt2_config.output_hidden_states = False
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if getattr(configs, 'prompt_domain', 0) and getattr(configs, 'content', None):
            self.description = configs.content
        else:
            descriptions = {
                'Wind': 'Multivariate wind measurements sampled at regular intervals.',
                'Exchange': 'Daily multivariate exchange rates.',
                'ILI': 'Weekly influenza-like illness observations.',
                'Weather': 'Multivariate weather measurements.',
            }
            self.description = descriptions.get(
                str(getattr(configs, 'data', '')),
                'Electricity transformer temperature and load time-series observations.')

        self.dropout = nn.Dropout(configs.dropout)

        # Separate patch paths preserve component identities before the one LLM.
        self.trend_patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.seasonal_patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.residual_patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = int(getattr(configs, "num_tokens", 1000))
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # PatchEmbedding pads the last patch; each component uses same count.
        self.component_patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.patch_nums = self.component_patch_nums * 3
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def _fit_ridge(self, y, design, penalty):
        """Batch-wise differentiable ridge fitting; compute the solve in FP32.

        y: (B, T, N); design: (T, D); penalty: (D,).
        """
        B, T, N = y.shape
        ctx = torch.cuda.amp.autocast(enabled=False) if y.is_cuda else nullcontext()
        with ctx:
            design = design.float()
            y2d = y.float().permute(1, 0, 2).reshape(T, B * N)
            xt = design.T
            lhs = xt @ design + torch.diag(penalty.float())
            rhs = xt @ y2d
            beta = torch.linalg.solve(lhs, rhs)
            fitted = (design @ beta).reshape(T, B, N).permute(1, 0, 2)
        return fitted.to(y.dtype)

    def _piecewise_trend(self, y):
        """Prophet-inspired piecewise-linear trend with fixed changepoints."""
        T = y.shape[1]
        t = torch.linspace(0, 1, T, device=y.device, dtype=torch.float32)
        columns = [torch.ones_like(t), t]
        num_cp = min(max(self.n_changepoints, 0), max(T - 2, 0))
        cp_end = min(max(self.changepoint_range, 0.0), 1.0)
        if num_cp:
            cps = torch.linspace(0, cp_end, num_cp + 2,
                                 device=y.device, dtype=torch.float32)[1:-1]
            columns.extend(torch.relu(t - cp) for cp in cps)
        design = torch.stack(columns, dim=1)
        penalty = torch.full((design.shape[1],), 1e-4,
                             device=y.device, dtype=torch.float32)
        if num_cp:
            penalty[2:] = 1.0 / (self.changepoint_prior_scale ** 2 + 1e-6)
        return self._fit_ridge(y, design, penalty)

    def _fourier_seasonal(self, y):
        """Fixed Fourier basis. Fit only its coefficients on observed history."""
        T = y.shape[1]
        t = torch.arange(T, device=y.device, dtype=torch.float32)
        columns = []
        # Avoid duplicate Fourier harmonics when period is a small integer.
        integer_period = int(round(self.seasonal_period))
        max_order = (integer_period - 1) // 2 if abs(self.seasonal_period - integer_period) < 1e-6 else self.fourier_order
        order = min(max(0, self.fourier_order), max_order)
        for k in range(1, order + 1):
            angle = 2 * pi * k * t / self.seasonal_period
            columns.extend([torch.sin(angle), torch.cos(angle)])
        # Even periods have a distinct Nyquist cosine. No duplicate sine.
        if integer_period % 2 == 0 and abs(self.seasonal_period - integer_period) < 1e-6 and self.fourier_order >= integer_period // 2:
            columns.append(torch.cos(pi * t))
        if not columns:
            return torch.zeros_like(y)
        design = torch.stack(columns, dim=1)
        penalty = torch.full((design.shape[1],),
                             1.0 / (self.seasonality_prior_scale ** 2 + 1e-6),
                             device=y.device, dtype=torch.float32)
        return self._fit_ridge(y, design, penalty)

    def decompose(self, normalized_x):
        """T + S + R reconstruct normalized input (no Event branch)."""
        trend = self._piecewise_trend(normalized_x)
        seasonal = self._fourier_seasonal(normalized_x - trend)
        residual = normalized_x - trend - seasonal
        return trend, seasonal, residual

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """dTimeLLM: RevIN -> PFD(T,S,R) -> combined component tokens -> one LLM."""
        x_norm = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_norm.shape
        if T != self.seq_len:
            raise ValueError(f'Expected seq_len={self.seq_len}, got {T}')
        trend, seasonal, residual = self.decompose(x_norm)

        global_flat = x_norm.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = global_flat.amin(dim=1).squeeze(-1)
        max_values = global_flat.amax(dim=1).squeeze(-1)
        medians = global_flat.median(dim=1).values.squeeze(-1)
        lags = self.calcute_lags(global_flat)
        diff = global_flat[:, -1, 0] - global_flat[:, 0, 0]
        prompts = []
        for idx in range(B * N):
            prompts.append(
                f'<|start_prompt|>Dataset description: {self.description} '
                f'Task: forecast the next {self.pred_len} steps from {self.seq_len} history steps. '
                'Input was decomposed using a changepoint-aware piecewise-linear trend '
                'and fixed Fourier-basis seasonality, with an additive residual; '
                'component tokens are combined before one LLM. '
                f'Input statistics: min {min_values[idx].item():.4f}, '
                f'max {max_values[idx].item():.4f}, '
                f'median {medians[idx].item():.4f}, '
                f'trend direction {"upward" if diff[idx].item() > 0 else "downward"}, '
                f'lags {lags[idx].tolist()}.<|end_prompt|>'
            )

        prompt_ids = self.tokenizer(prompts, return_tensors='pt', padding=True,
                                    truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt_ids.to(x_norm.device))
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)).permute(1, 0)

        def component_tokens(z, embed):
            # (B, T, N) -> (B, N, T); PatchEmbedding returns (B*N, P, d_model).
            z_tokens, nvars = embed(z.permute(0, 2, 1).contiguous().float())
            return self.reprogramming_layer(
                z_tokens, source_embeddings, source_embeddings), nvars

        trend_tokens, n_vars = component_tokens(trend, self.trend_patch_embedding)
        seasonal_tokens, _ = component_tokens(seasonal, self.seasonal_patch_embedding)
        residual_tokens, _ = component_tokens(residual, self.residual_patch_embedding)

        # Structural information survives: concatenation, not sum (which equals x).
        llm_in = torch.cat([
            prompt_embeddings, trend_tokens, seasonal_tokens, residual_tokens
        ], dim=1)
        llm_out = self.llm_model(inputs_embeds=llm_in).last_hidden_state
        llm_out = llm_out[:, :, :self.d_ff]
        llm_out = llm_out.reshape(B, n_vars, llm_out.shape[1], self.d_ff)
        llm_out = llm_out.permute(0, 1, 3, 2).contiguous()
        # Use all T/S/R patch tokens, excluding only the prompt tokens.
        llm_out = llm_out[:, :, :, -self.patch_nums:]
        dec_out = self.output_projection(llm_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        return self.normalize_layers(dec_out, 'denorm')

    def calcute_lags(self, x_enc):
        # FP32 FFT works for non-power-of-two lengths such as ILI seq_len=36.
        series = x_enc.permute(0, 2, 1).contiguous().float()
        spectrum = torch.fft.rfft(series, dim=-1)
        corr = torch.fft.irfft(spectrum * torch.conj(spectrum),
                               n=series.shape[-1], dim=-1)
        mean_value = corr.mean(dim=1)
        k = min(self.top_k, mean_value.shape[-1])
        _, lags = torch.topk(mean_value, k, dim=-1)
        if k < self.top_k:
            lags = torch.nn.functional.pad(lags, (0, self.top_k - k))
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
