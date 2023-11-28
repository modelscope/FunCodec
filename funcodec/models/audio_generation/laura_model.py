import logging
from typing import Any, List, Tuple, Dict, Optional, Union
import torch
import torch.nn as nn
from funcodec.modules.embedding import PositionalEncoding, ScaledPositionalEncoding
from funcodec.modules.nets_utils import (
    subsequent_mask, make_pad_mask, th_accuracy, pad_list
)
from funcodec.train.abs_espnet_model import AbsESPnetModel
import torch.nn.functional as F
from funcodec.torch_utils.device_funcs import force_gatherable
from funcodec.losses.label_smoothing_loss import LabelSmoothingLoss
from copy import deepcopy


class QuantizerCodebook(torch.nn.Module):
    def __init__(
            self,
            num_quantizers,
            codebook_size,
            codebook_dim
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)
        self.register_buffer("embed", embed)
        codec_index_shift = 1024 * torch.arange(32, dtype=torch.float32)[None, None, :]
        self.register_buffer("codec_index_shift", codec_index_shift)

    def save_embedding(self, file_name, dense_emb, emb_lengths):
        import kaldiio
        wav_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(file_name, file_name))
        dense_emb = dense_emb.cpu().numpy()
        for i in range(min(dense_emb.shape[0], 10)):
            wav_writer(str(i), dense_emb[i, :emb_lengths[i]])

        wav_writer.close()

    def forward(self, codec: torch.Tensor, codec_lengths, return_subs=False):
        if len(codec.shape) == 2:
            codec = codec.unsqueeze(-1)
        bz, tt, nq = codec.shape[0], codec.shape[1], codec.shape[2]
        codec_mask = ~make_pad_mask(codec_lengths, maxlen=codec.shape[1]).unsqueeze(-1).to(codec.device)
        codec = codec * codec_mask + self.codec_index_shift[:, :, :nq].long()
        codec = codec.reshape(-1, nq)
        emb = self.embed.reshape(-1, self.codebook_dim)
        codec_emb = F.embedding(codec, emb)  # (BT, Nq, D)
        dense_emb = codec_emb.sum(dim=1)
        dense_emb = dense_emb.reshape(bz, tt, self.codebook_dim)
        if return_subs:
            sub_embs = codec_emb.reshape(bz, tt, nq, self.codebook_dim) * codec_mask.unsqueeze(-2)
            return dense_emb * codec_mask, sub_embs
        return dense_emb * codec_mask


class LauraGenModel(AbsESPnetModel):
    """
    This class implement the LauraGPT-style audio generation model [1]. It can be trained for
    speech, music, audio generation tasks with corresponding datasets.

    [1] LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT, 2023,
    https://arxiv.org/abs/2310.04673
    """
    def __init__(
            self,
            input_size,                     # seq size of text embeddings
            text_encoder: nn.Module,        # encode text inputs
            codec_encoder: nn.Module,       # predict codec_emb according to codec_1st
            vocab_size: int = 0,            # 0 for embedding inputs, > 0 for token inputs such as phoneme
            token_list: List[str] = None,   # None for embedding inputs, not None for token inputs
            pos_enc: str = "abs_pos",
            codec_conf: Dict = None,
            ignore_id: int = -1,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.1,
            codec_lm_conf: Dict = None,
            codec_sampling_ratio: float = 0.0,
            predict_nq: int = 1,
            pos_emb_type: str = "split",
    ):
        super().__init__()
        if pos_enc in ["sinusoidal", "abs_pos"]:
            pos_enc_class = PositionalEncoding
        elif pos_enc == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc is None:
            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        assert pos_emb_type in ["split", "uni"], f"pos_emb_type must be split or uni rather than {pos_emb_type}"

        self.ignore_id = ignore_id
        self.codec_sampling_ratio = codec_sampling_ratio
        self.num_quantizers = num_quantizers = codec_conf.get("num_quantizers", 32)
        self.codebook_size = codebook_size = codec_conf.get("codebook_size", 1024)
        self.codebook_dim = codebook_dim = codec_conf.get("codebook_dim", 128)
        self.predict_nq = predict_nq
        self.pos_emb_func = pos_enc_class(self.codebook_dim, 0.1)
        self.pos_emb_type = pos_emb_type

        # 1. build text inputs related modules
        self.text_encoder = text_encoder
        self.text_enc_out_layer = nn.Linear(
            self.text_encoder.output_size() if text_encoder is not None else input_size,
            self.codebook_dim
        )
        self.vocab_size = vocab_size
        self.token_list = token_list
        if vocab_size > 0:
            self.token_embedding = torch.nn.Embedding(vocab_size, input_size)

        # 2. build Music language model related moduels
        self.sos_eos = 0
        self.task_id = 1
        # embedding for sos_eos and task id
        self.lm_embedding = torch.nn.Embedding(2, self.codebook_dim)
        self.lm_out_voc_size = (self.codebook_size + 1) * self.predict_nq
        self.codec_lm = self.build_codec_lm(codec_lm_conf)

        # 3. build fine codec predictor
        self.codec_encoder = codec_encoder
        self.codec_encoder_out_layer = nn.Linear(codec_encoder.output_size(), self.codebook_dim)

        self.quantizer_codebook = QuantizerCodebook(num_quantizers, codebook_size, codebook_dim)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.lm_out_voc_size // self.predict_nq,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
            reduction=False,
        )
        self.length_normalized_loss = length_normalized_loss
        from funcodec.models.quantizer.costume_quantizer import CostumeQuantizer
        self.quantizer = CostumeQuantizer(
            input_size=self.codebook_dim,
            codebook_size=self.codebook_size,
            num_quantizers=32,
            ema_decay=0.99,
            kmeans_init=True,
            sampling_rate=16000,
            quantize_dropout=False,
            use_ddp=True,
        )

    def build_codec_lm(self, conf: Dict):
        name = conf.pop("name")
        if name == "transformer":
            from funcodec.lm.transformer_lm import TransformerEmbedLM
            if "text_vocab_size" in conf:
                lm_model = TransformerEmbedLM(
                    vocab_size=self.lm_out_voc_size,
                    **conf
                )
            else:
                lm_model = TransformerEmbedLM(
                    vocab_size=self.lm_out_voc_size,
                    text_vocab_size=self.lm_out_voc_size,
                    **conf
                )
        else:
            raise TypeError(f"Unknown codec decoder type {name}")

        conf["name"] = name
        return lm_model

    def _target_mask(self, lengths):
        ys_mask = ~make_pad_mask(lengths)
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        if self.text_encoder is not None:
            outs, out_lens, _ = self.text_encoder(text, text_lengths)
            outs = self.text_enc_out_layer(outs)
        else:
            if text.shape[-1] == self.codebook_dim:
                outs, out_lens = text, text_lengths
            else:
                outs = self.text_enc_out_layer(text)
                out_lens = text_lengths

        return outs, out_lens

    def build_llm_io(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: Optional[torch.Tensor] = None,
            codec_lengths: Optional[torch.Tensor] = None,
            need_targets: bool = True,
    ):
        """build inputs and targets for language model

                Normally, this function is called in batchify_nll.
                Args:
                    text: (Batch, Length, Dim)
                    text_lengths: (Batch,)
                    codec: (Batch, Length)
                    codec_lengths: (Batch,)
                    need_targets: bool, whether provide targets
                """

        if need_targets:
            assert codec is not None and codec_lengths is not None, \
                "need_target=True, but codec or codec_length is None"

        sos_eos_emb = self.lm_embedding(torch.tensor([self.sos_eos], dtype=torch.int64, device=text.device))
        task_id_emb = self.lm_embedding(torch.tensor([self.task_id], dtype=torch.int64, device=text.device))
        codec_emb = None
        if codec is not None and codec_lengths is not None:
            codec_emb = self.calc_dense_vector(codec, codec_lengths)
        inputs_list = []
        for i, text_len in enumerate(text_lengths):
            one_input = [sos_eos_emb, text[i, :text_len], task_id_emb]
            if codec_emb is not None:
                one_input.append(codec_emb[i, :codec_lengths[i]])
            inputs_list.append(torch.cat(one_input, dim=0))
        llm_inputs = pad_list(inputs_list, 0.0)
        llm_lengths = text_lengths + 2
        if codec_emb is not None:
            llm_lengths = llm_lengths + codec_lengths

        if not need_targets:
            return llm_inputs, llm_lengths

        bb, tt = text.shape[0], codec_lengths.max() + 1
        llm_targets = torch.zeros([bb, tt, self.predict_nq], dtype=torch.int64, device=text.device)
        for i, codec_len in enumerate(codec_lengths):
            llm_targets[i, :codec_len] = codec[i, :codec_len]
            llm_targets[i, codec_len] = self.codebook_size + self.sos_eos

        return (llm_inputs, llm_targets), (llm_lengths, codec_lengths + 1)

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        codec: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length, Dim)
            text_lengths: (Batch,)
            codec: (Batch, Length)
            codec_lengths: (Batch,)
        """
        batch_size = text.size(0)
        # For data parallel
        text = text[:, :text_lengths.max()]
        codec = codec[:, :codec_lengths.max()]

        # build inputs and targets for language model
        (sequence, target), (x_lengths, y_lengths) = self.build_llm_io(
            text, text_lengths,
            codec, codec_lengths,
            need_targets=True
        )

        # 2a. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        sequence = sequence[:, :x_lengths.max()]
        target = target[:, :y_lengths.max()]
        y, _ = self.codec_lm(sequence, x_lengths, text_lengths+1)
        bb, tt = y.shape[0], y.shape[1]
        y = y.reshape(bb, tt, self.predict_nq, -1)
        # 2b. Extract real logits
        logits_list = []
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            logits_list.append(y[i, text_len + 1:text_len + 2 + codec_len])
        logits = pad_list(logits_list, 0.0)

        # 3. Calc negative log likelihood
        tt = logits.shape[1]
        nll = self.criterion_ce(
            logits.reshape(bb, tt * self.predict_nq, -1),
            target.reshape(bb, tt * self.predict_nq)
        )
        nll = nll.sum(-1)
        # nll: (BxL,) -> (BxL,)
        nll.masked_fill_(make_pad_mask(y_lengths * self.predict_nq).to(nll.device).view(-1), 0.0)
        # nll: (BxL,) -> (B, L)
        nll = nll.reshape(batch_size, -1).reshape(batch_size, tt, self.predict_nq)

        return nll, logits, target, codec_lengths+1

    def cal_codec_emb(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec_prob: torch.Tensor,
            codec_lengths: torch.Tensor,
    ):
        first_nq_emb = None
        for i in range(self.predict_nq):
            one_emb = torch.matmul(codec_prob[:, :, i], self.quantizer_codebook.embed[i:i+1].detach())
            if first_nq_emb is None:
                first_nq_emb = one_emb
            else:
                first_nq_emb = first_nq_emb + one_emb

        model_inputs = []
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            if self.pos_emb_type == "split":
                one_in = [
                    self.pos_emb_func(text[i:i+1, :text_len]).squeeze(0),
                    self.pos_emb_func(first_nq_emb[i:i+1, :codec_len]).squeeze(0)
                ]
            else:
                one_in = [text[i, :text_len], first_nq_emb[i, :codec_len]]
            model_inputs.append(torch.cat(one_in, dim=0))
        model_input_lengths = text_lengths + codec_lengths
        model_inputs = pad_list(model_inputs, 0.0)
        model_inputs = model_inputs[:, :model_input_lengths.max()]

        model_outs, model_outs_lens, _ = self.codec_encoder(model_inputs, model_input_lengths)
        model_outs = self.codec_encoder_out_layer(model_outs)

        outs = torch.zeros([text.shape[0], codec_lengths.max(), self.codebook_dim], requires_grad=True).to(text)
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            outs[i, :codec_len] = model_outs[i, text_len: text_len+codec_len]

        return outs, codec_lengths

    def calc_reg_loss(self, prediction, target, length):
        loss_mask = ~make_pad_mask(length, target)
        l1_loss = F.l1_loss(prediction, target, reduction="none")
        l1_loss = (l1_loss * loss_mask).sum() / loss_mask.sum()
        l2_loss = 0.5 * F.mse_loss(prediction, target, reduction="none")
        l2_loss = (l2_loss * loss_mask).sum() / loss_mask.sum()

        return l1_loss * 0.5 + l2_loss * 0.5, l1_loss, l2_loss

    def calc_dense_vector(self, codec, codec_lengths):
        """
        Args:
            codec: (B, T, Nq)
            codec_lengths: (B, )
        """
        with torch.no_grad():
            return self.quantizer_codebook(codec, codec_lengths)

    def prob_sampler(
            self,
            logits: torch.Tensor,
            codec: torch.Tensor,
            codec_lengths: torch.Tensor,
    ):
        """ Sampling ground-truth prob to replace wrongly predicted prob
        Args:
            logits: (B, T, N, V)
            codec: (B, T, N)
            codec_lengths: (B,)
        """
        assert logits.shape[1] == codec.shape[1], \
            f"lengths of logits and codec mismatch: {logits.shape[1]} and {codec.shape[1]}"
        bb, tt = logits.shape[0], logits.shape[1]
        valid_mask = (~make_pad_mask(codec_lengths)).view(bb, tt, 1, 1).to(logits.device)

        soft_prob = torch.softmax(logits, dim=-1)
        pred_token = torch.argmax(soft_prob, dim=-1)
        hard_prob = F.one_hot(pred_token, self.codebook_size).float()
        # go-through gradient estimation
        pred_prob = soft_prob + (hard_prob - soft_prob).detach()
        if self.codec_sampling_ratio == 0.0:
            return pred_prob * valid_mask

        gt_prob = F.one_hot(
            torch.clamp(codec, 0, self.codebook_size - 1),
            self.codebook_size
        ).float()
        if self.codec_sampling_ratio == 1.0:
            return gt_prob * valid_mask

        # bb, tt, nn
        correct_mask = (pred_token == codec)
        # higher codec_sampling_ratio means less prediction usage
        sampling_mask = torch.rand_like(correct_mask.float()) > self.codec_sampling_ratio
        # for correct tokens or (wrong tokens without sampling), we use predictions
        input_mask = (torch.logical_or(
            correct_mask,
            torch.logical_and(~correct_mask, sampling_mask))
        ).unsqueeze(-1)
        prob = input_mask * pred_prob + (~input_mask) * gt_prob

        # masking out the padding part
        return prob * valid_mask

    def forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: torch.Tensor,
            codec_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            codec: (B, T, N)
            codec_lengths: (B,)
        """
        text = text[:, :text_lengths.max()]
        codec = codec[:, :codec_lengths.max()].long()
        if self.vocab_size > 0:
            mask = text != self.ignore_id
            text = self.token_embedding(text * mask) * mask.unsqueeze(-1)
        # 1. encode text
        text, text_lengths = self.encode(text, text_lengths)

        # 2. generate the first `predict_nq` codec groups
        nll, logits, target, target_lengths = self.nll(text, text_lengths, codec[:, :, :self.predict_nq], codec_lengths)
        output_mask = ~make_pad_mask(target_lengths, maxlen=target_lengths.max()).to(text.device).unsqueeze(-1)
        total, batch_size = output_mask.sum() * self.predict_nq, nll.shape[0] * self.predict_nq
        denom = total if self.length_normalized_loss else batch_size
        nll_loss = (nll * output_mask).sum() / denom

        # 3. generate dense codec vectors
        # sampling codec prob
        prob = self.prob_sampler(
            # remove <eos> from logits
            logits[:, :-1, :self.predict_nq, :self.codebook_size],
            codec[:, :, :self.predict_nq],
            codec_lengths
        )
        codec_emb, codec_emb_lens = self.cal_codec_emb(text, text_lengths, prob, codec_lengths)

        # 4. loss calculation
        target_emb = self.calc_dense_vector(codec, codec_lengths)
        reg_loss, l1_loss, l2_loss = self.calc_reg_loss(codec_emb, target_emb, codec_lengths)
        loss = reg_loss + nll_loss
        stats = dict(
            loss=loss.detach(),
            nll_loss=nll_loss.detach(),
            reg_loss=reg_loss.detach(),
            reg_l1_loss=l1_loss.detach(),
            reg_l2_loss=l2_loss.detach(),
            batch_size=text.shape[0],
            seq_length=text_lengths.max() + codec_lengths.max(),
        )

        # 5. accuracy calculation
        with torch.no_grad():
            cc = logits.shape[-1]
            for i in range(self.predict_nq):
                acc = th_accuracy(
                    logits[:, :, i, :].reshape(-1, cc),
                    target[:, :, i],
                    self.ignore_id
                )
                stats[f"out_acc_{i+1}"] = acc

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            sampling: Union[bool, int, float] = True,
            beam_size: int = 1,
    ):
        if isinstance(sampling, bool):
            if sampling:
                top_ids = weighted_scores.softmax(dim=0).multinomial(beam_size, replacement=True)
            else:
                top_ids = weighted_scores.topk(beam_size)[1]
        elif isinstance(sampling, int):
            prob, indices = weighted_scores.softmax(dim=0).topk(sampling)
            sampling_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, float):
            prob, indices = [], []
            cum_prob = 0.0
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            for i in range(len(sorted_idx)):
                if cum_prob < sampling:
                    cum_prob += sorted_value[i]
                    prob.append(sorted_value[i])
                    indices.append(sorted_idx[i])
                else:
                    break
            prob = torch.tensor(prob).to(weighted_scores)
            indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
            sampling_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[sampling_ids]
        else:
            raise NotImplementedError(f"Not implemented for {type(sampling)} sampling")

        return top_ids

    def decode_codec(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            max_length: int = 30 * 25,
            sampling: Union[bool, int, float] = True,
            beam_size: int = 1,
            continual: List = None,
    ) -> torch.Tensor:
        device = text.device
        out_tokens = [] if continual is None else deepcopy(continual)
        sos_eos_emb = self.lm_embedding(torch.tensor([[self.sos_eos]], dtype=torch.int64, device=device))
        task_id_emb = self.lm_embedding(torch.tensor([[self.task_id]], dtype=torch.int64, device=device))
        prompt = torch.cat([sos_eos_emb, text, task_id_emb], dim=1)
        state = None
        for i in range(max_length):
            if len(out_tokens) > 0:
                codec_prompt = torch.tensor([out_tokens], dtype=torch.int64, device=device)
                codec_lengths = torch.tensor([len(out_tokens)], dtype=torch.int64, device=device)
                # if any quantizer output is eos
                if torch.any(codec_prompt[:, -1] == (self.codebook_size+self.sos_eos)):
                    break
                seq_input, _ = self.build_llm_io(
                    text, text_lengths,
                    codec_prompt, codec_lengths,
                    need_targets=False
                )
            else:
                seq_input, _ = self.build_llm_io(
                    text, text_lengths, None, None,
                    need_targets=False
                )

            # not use state, since has not aligned
            pred, _ = self.codec_lm.score(seq_input[0], state, prompt[0])

            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            top_ids = []
            for k in range(self.predict_nq):
                top_ids.append(self.sampling_ids(pred[k], sampling, beam_size)[0].item())
            out_tokens.append(top_ids)

        # remove eos token
        if torch.any(torch.tensor(out_tokens[-1], dtype=torch.int64) == self.codebook_size+self.sos_eos):
            out_tokens = out_tokens[:-1]

        return torch.tensor([out_tokens], dtype=torch.int64, device=device)

    def syn_audio(
            self,
            codec: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec_model,
            continual_length=None,
    ):
        codec = codec[:, :, :self.predict_nq]
        prob = F.one_hot(
            torch.clamp(codec, 0, self.codebook_size-1),
            self.codebook_size
        ).float()
        codec_lengths = torch.tensor([codec.shape[1]], dtype=torch.int64, device=text.device)
        codec_emb, codec_emb_lens = self.cal_codec_emb(text, text_lengths, prob, codec_lengths)
        _, _, recon_wav, _ = codec_model(codec_emb[:, continual_length:], run_mod="decode_emb")

        return recon_wav

    def collect_feats(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: torch.Tensor,
            codec_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        feats, feats_lengths = codec, codec_lengths

        return {"feats": feats, "feats_lengths": feats_lengths}
