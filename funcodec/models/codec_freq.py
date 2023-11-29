# Copyright 2023 Zhihao Du
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-End Speech Tokenizer SoundStream."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple, List, Union
import typing as tp
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types
from funcodec.train.abs_gan_espnet_model import AbsGANESPnetModel
from funcodec.torch_utils.device_funcs import force_gatherable
from librosa.filters import mel as librosa_mel_fn
from funcodec.losses.label_smoothing_loss import LabelSmoothingLoss
from funcodec.layers.mask_along_axis import MaskAlongAxisVariableMaxWidth
import logging


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cpu'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audioin, return_power_spec=False):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        power_spec = torch.sum(torch.pow(fft, 2), dim=[-1])
        mel_output = torch.matmul(self.mel_basis, power_spec)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        if return_power_spec:
            log_power_spec = torch.log10(torch.clamp(power_spec, min=1e-5))
            return log_mel_spec, log_power_spec
        return log_mel_spec


EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class FreqCodec(AbsGANESPnetModel):
    """Encodec (generator + discriminator).

    This is the Encodec model
    """

    def __init__(
            self,
            input_size: int,
            odim: int = 512,
            frontend: torch.nn.Module = None,
            encoder: torch.nn.Module = None,
            quantizer: torch.nn.Module = None,
            decoder: torch.nn.Module = None,
            discriminator: Optional[torch.nn.Module] = None,
            target_sample_hz: int = 24_000,
            multi_spectral_window_powers_of_two: Union[Tuple, List] = tuple(range(5, 11)),
            multi_spectral_n_mels: int = 64,
            recon_loss_weight: float = 1.,
            multi_spectral_recon_loss_weight: float = 1.,
            adversarial_loss_weight: float = 1/9,
            feat_match_loss_weight: float = 100/9,
            enc_quant_loss_weight: float = 1.0,
            audio_normalize: bool = False,
            segment_dur: Optional[float] = 1.0,
            overlap_ratio: Optional[float] = 0.01,
            use_power_spec_loss: Optional[bool] = False,
            bypass_quantizer: bool = False,
            codec_domain: List = ("time", "time"),
            domain_conf: Optional[Dict] = {},
            phase_invariant_training: bool = False,
            # pit means phase invariant training
            pit_feat_loss_weight: float = 1,
            pit_disc_loss_weight: float = 1000,
            feat_match_layer_start: int = -1,
    ):
        """Initialize SoundStream model.

        Args:
            input_size: the channel or dimension of input data
            odim: the dimension of model
            encoder: encoder
            quantizer: quantizer
            decoder: decoder
            discriminators: several discriminators, such as STFTDisc, MultiScaleDisc, MultiPeriodDisc
            discr_multi_scales: time scales of multiple discriminators
            stft_normalized: whether to normalize by magnitude after STFT, default: False.
            multi_spectral_window_powers_of_two: for multiple spectral recon loss
            multi_spectral_n_ffts: fft bins
            multi_spectral_n_mels: Mel frequency bins
            recon_loss_weight: the weight of time-domain reconstruction loss
            multi_spectral_recon_loss_weight: the weight of frequency-domain reconstruction loss
            adversarial_loss_weight: the weight of adversarial loss from discriminator
            feat_match_loss_weight: the weight of intermediate feature loss from discriminator
            cache_generator_outputs: Whether to cache generator outputs.
        """
        assert check_argument_types()
        super().__init__()

        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        # Used by task and trainer
        self.gen_model_list = [self.encoder, self.quantizer, self.decoder]
        self.discriminator = discriminator
        self.bypass_quantizer = bypass_quantizer
        self.codec_domain = codec_domain
        self.domain_conf = domain_conf
        if codec_domain[0] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=None,
            )
        elif codec_domain[0] in ["mag"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=1,
            )
        elif codec_domain[0] == "mel":
            self.enc_trans_func = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_hz,
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                n_mels=80,
                power=2,
            )
        if codec_domain[1] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.dec_trans_func = torchaudio.transforms.InverseSpectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
            )

        # multi spectral reconstruction
        self.mel_spec_transforms = nn.ModuleList([])

        for powers in multi_spectral_window_powers_of_two:
            win_length = 2 ** powers

            melspec_transform = Audio2Mel(
                sampling_rate=target_sample_hz,
                win_length=win_length,
                hop_length=win_length // 4,
                n_mel_channels=multi_spectral_n_mels
            )

            self.mel_spec_transforms.append(melspec_transform)

        # loss weights
        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feat_match_loss_weight = feat_match_loss_weight
        self.enc_quant_loss_weight = enc_quant_loss_weight
        self.register_buffer('zero', torch.tensor([0.]), persistent=False)
        self.gen_loss = 0
        self.audio_normalize = audio_normalize
        self.segment_dur = segment_dur
        self.overlap_ratio = overlap_ratio
        self.sample_rate = target_sample_hz
        self.forward_step = 0
        self.use_power_spec_loss = use_power_spec_loss

        self.feat_match_layer_start = feat_match_layer_start
        self.phaseaug = None
        self.phase_invariant_training = phase_invariant_training
        self.pit_feat_loss_weight = pit_feat_loss_weight
        self.pit_disc_loss_weight = pit_disc_loss_weight
        if phase_invariant_training:
            from phaseaug.phaseaug import PhaseAug
            self.phaseaug = PhaseAug(nfft=512, hop=160)

    @property
    def generator(self):
        return torch.nn.ModuleList(self.gen_model_list)


    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment_dur is None:
            return None
        return int(self.segment_dur * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap_ratio) * segment_length))

    def forward(
        self,
        forward_generator: bool = True,
        batch: Dict = None,
    ) -> Dict[str, Any]:
        """Forward functions of generator and discriminator.

        Args:
            forward_generator (bool): Whether to forward generator.
            batch (Dict[str, Tensor]): one batch including:
                speech (Tensor): Speech waveform tensor (B, T_wav).
                speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            if self.training:
                self.forward_step += 1
            return self._forward_generator(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
            )
        else:
            return self._forward_discriminator(
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
            )

    def _encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert 0 < channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        # print("length:", length, "stride:", stride)
        for offset in range(0, length, stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment_dur is None or duration <= 1e-5 + self.segment_dur

        if self.audio_normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        if self.codec_domain[0] == "stft":
            x_complex = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 2:
                x = torch.stack([x_complex.real, x_complex.imag], dim=1)
            else:
                x = torch.cat([x_complex.real, x_complex.imag], dim=1)
        elif self.codec_domain[0] == "mag":
            x_mag = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 1:
                x = x_mag.unsqueeze(1)
            else:
                x = x_mag
        elif self.codec_domain[0] == "mag_angle":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_angle = torch.angle(x_complex)
            if self.encoder.input_size == 2:
                x = torch.stack([x_log_mag, x_angle], dim=1)
            else:
                x = torch.cat([x_log_mag, x_angle], dim=1)
        elif self.codec_domain[0] == "mag_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_phase = x_complex / torch.clamp(x_mag, min=1e-6)
            if self.encoder.input_size == 3:
                x = torch.stack([x_log_mag, x_phase.real, x_phase.imag], dim=1)
            else:
                x = torch.cat([x_log_mag, x_phase.real, x_phase.imag], dim=1)
        elif self.codec_domain[0] == "mel":
            x = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 1:
                x = x.unsqueeze(1)
        elif self.codec_domain[0] == "mag_oracle_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x = torch.abs(x_complex)
            if self.encoder.input_size == 1:
                x = x.unsqueeze(1)
            x_phase = torch.angle(x_complex)
            scale = (scale, x_phase)

        emb = self.encoder(x)

        return emb, scale

    def _decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = []
        for frame in encoded_frames:
            frames.append(self._decode_frame(frame))

        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        emb = codes
        out = self.decoder(emb)
        if self.codec_domain[1] == "stft":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1]//2, dim=1)
            else:
                out_list = torch.split(out, 1, dim=1)
            out = torch.complex(out_list[0], out_list[1])
            out = self.dec_trans_func(out).unsqueeze(1)
        elif self.codec_domain[1] == "mag_phase":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1] // 3, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(out, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_phase = torch.complex(out_list[1], out_list[2])
            out = x_mag * x_phase
            out = self.dec_trans_func(out).unsqueeze(1)
        elif self.codec_domain[1] == "mag_angle":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1] // 2, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(out, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_angle = torch.sin(out_list[1]) * torch.pi
            x_spec = torch.complex(torch.cos(x_angle) * x_mag, torch.sin(x_angle) * x_mag)
            out = self.dec_trans_func(x_spec).unsqueeze(1)
        elif self.codec_domain[1] == "mag_oracle_phase":
            if len(out.shape) == 4:
                out = out.squeeze(1)
            (scale, x_angle), x_mag = scale, out
            x_spec = torch.complex(torch.cos(x_angle)*x_mag, torch.sin(x_angle)*x_mag)
            out = self.dec_trans_func(x_spec).unsqueeze(1)
        elif (self.codec_domain[0] in ["stft", "mag", "mag_phase", "mag_angle", "mag_oracle_phase"] and
              self.codec_domain[1] == "time"):
            hop_length = self.domain_conf.get("hop_length", 160)
            out = out[:, :, hop_length//2: -hop_length//2]

        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()

        l1Loss = torch.nn.L1Loss(reduction='mean')
        l2Loss = torch.nn.MSELoss(reduction='mean')
        commit_losses = []
        enc_quant_losses = []
        codes = []
        frames = self._encode(speech)
        for emb, scale in frames:
            quant_in = emb
            quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in)
            code_embs = quant_out
            # qv = self.quantizer.forward(emb, self.sample_rate, self.bandwidth)
            commit_losses.append(commit_loss)
            enc_quant_losses.append(l2Loss(quant_out, quant_in) ** 2)
            codes.append((code_embs, scale))
        recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        commit_loss = torch.stack(commit_losses).sum()
        enc_quant_loss = torch.stack(enc_quant_losses).sum()

        # A: recon loss
        recon_loss = l1Loss(orig_speech, recon_speech)
        # B: multiple spectral recon loss - eq (4) and (5) in https://arxiv.org/abs/2107.03312
        multi_spectral_recon_loss = self.zero
        if self.multi_spectral_recon_loss_weight > 0:
            for mel_transform in self.mel_spec_transforms:
                # mel_transform: (..., Time) -> (..., n_mel, Frame)
                if not self.use_power_spec_loss:
                    orig_mel, recon_mel = map(mel_transform, (orig_speech, recon_speech))

                    l1_mel_loss = l1Loss(orig_mel, recon_mel)
                    l2_mel_loss = l2Loss(orig_mel, recon_mel)
                else:
                    orig_mel, orig_power = mel_transform(orig_speech, self.use_power_spec_loss)
                    recon_mel, recon_power = mel_transform(recon_speech, self.use_power_spec_loss)
                    l1_mel_loss = l1Loss(orig_mel, recon_mel) * 0.5 + l1Loss(orig_power, recon_power) * 0.5
                    l2_mel_loss = l2Loss(orig_mel, recon_mel) * 0.5 + l2Loss(orig_power, recon_power) * 0.5

                multi_spectral_recon_loss = multi_spectral_recon_loss + (l1_mel_loss + l2_mel_loss)

            multi_spectral_recon_loss = multi_spectral_recon_loss / len(self.mel_spec_transforms)
        # C-1: calculate discriminator outputs
        # disc_outputs in the format [disc1_outputs, disc2_outputs, ...]
        # disc1_outputs includes [logits, intermediates]
        # intermediates includes [layer_1_intermediate, layer_2_intermediate, ...]
        fake_disc_outputs = self.discriminator(recon_speech)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            real_disc_outputs = self.discriminator(orig_speech)

        # C-2: calculate discriminator loss including adversarial and feat matching losses
        adversarial_losses = []
        disc_feature_losses = []
        for real_output, fake_output in zip(real_disc_outputs, fake_disc_outputs):
            real_logits, real_intermediates = real_output
            fake_logits, fake_intermediates = fake_output
            adversarial_losses.append(torch.mean(F.relu(1 - fake_logits)))
            for i, (real_inter, fake_inter) in enumerate(zip(real_intermediates, fake_intermediates)):
                if i >= self.feat_match_layer_start:
                    _loss = F.l1_loss(real_inter.detach(), fake_inter)
                    disc_feature_losses.append(_loss)

        adversarial_loss = torch.stack(adversarial_losses).mean()
        feat_match_loss = torch.stack(disc_feature_losses).mean()

        # calculate losses
        gen_loss = recon_loss * self.recon_loss_weight + \
                   multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight + \
                   adversarial_loss * self.adversarial_loss_weight + \
                   feat_match_loss * self.feat_match_loss_weight
        self.gen_loss += gen_loss.item()
        loss = (gen_loss + commit_loss +
                enc_quant_loss * self.enc_quant_loss_weight)

        stats = dict(
            generator_loss=loss.item(),
            generator_recon_loss=recon_loss.item(),
            generator_multi_spectral_recon_loss=multi_spectral_recon_loss.item(),
            generator_adv_loss=adversarial_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
            generator_commit_loss=commit_loss.item(),
            generator_enc_quant_loss=enc_quant_loss.item(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
            "real": orig_speech,
            "fake": recon_speech,
        }

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).
        """
        # setup
        batch_size = speech.size(0)
        speech = speech.unsqueeze(1)
        orig_speech = speech.clone()

        codes = []
        frames = self._encode(speech)
        for emb, scale in frames:
            quant_in = emb
            quant_out, indices, commit_loss, sub_quants = self.quantizer(quant_in)
            code_embs = quant_out
            codes.append((code_embs, scale))
        recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]

        # B: calculate discriminator outputs
        real, fake = orig_speech.clone(), recon_speech.detach()
        real_disc_outputs = self.discriminator(real)
        fake_disc_outputs = self.discriminator(fake)

        # C: calculate discriminator losses
        disc_losses = []
        for real_output, fake_output in zip(real_disc_outputs, fake_disc_outputs):
            real_logits, real_intermediates = real_output
            fake_logits, fake_intermediates = fake_output
            one_disc_loss = torch.mean(F.relu(1-real_logits)) + torch.mean(F.relu(1+fake_logits))
            disc_losses.append(one_disc_loss)
        disc_loss = torch.stack(disc_losses).mean()

        # Optional D: phase invariant training
        pit_disc_loss = self.zero
        if self.phase_invariant_training and self.phaseaug is not None:
            real_aug = self.phaseaug.forward(speech).detach()
            aug_disc_outputs = self.discriminator(real_aug)
            pit_disc_losses = []
            for real_output, aug_output in zip(real_disc_outputs, aug_disc_outputs):
                real_logits, real_intermediates = real_output
                aug_logits, aug_intermediates = aug_output
                one_pit_disc_loss = F.l1_loss(real_logits, aug_logits)

                disc_feature_losses = []
                for i, (real_inter, aug_inter) in enumerate(zip(real_intermediates, aug_intermediates)):
                    if i >= self.feat_match_layer_start:
                        _loss = F.l1_loss(real_inter, aug_inter)
                        disc_feature_losses.append(_loss)
                disc_feature_loss = torch.stack(disc_feature_losses).mean()

                one_pit_disc_loss = (one_pit_disc_loss +
                                     disc_feature_loss * self.pit_feat_loss_weight)

                pit_disc_losses.append(one_pit_disc_loss)
            pit_disc_loss = torch.stack(pit_disc_losses).mean()

        # To avoid discriminator overpowers the generator, without this recon losses may not converge
        if self.training:
            loss_mask = (disc_loss > self.gen_loss).float()
            disc_loss = disc_loss * loss_mask
            pit_disc_loss = pit_disc_loss * loss_mask

            if disc_loss.item() > self.gen_loss:
                logging.info(f"Will update discriminator: forward_step={self.forward_step}, "
                             f"disc_loss={disc_loss.item():.4f}, gen_loss={self.gen_loss:.4f}")
        self.gen_loss = 0

        # D: whether to use gradient penalty loss
        loss = disc_loss + pit_disc_loss * self.pit_disc_loss_weight

        stats = dict(
            discriminator_total_loss=loss.item(),
            discriminator_loss=disc_loss.item(),
        )
        if self.phase_invariant_training and self.phaseaug is not None:
            stats["pit_disc_loss"] = pit_disc_loss
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
            "real": orig_speech,
            "fake": recon_speech,
        }

    def inference(
            self,
            speech: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            speech (torch.Tensor): input speech
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = []
        code_idxs = []
        all_sub_quants = []
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        frames = self._encode(speech)
        for emb, scale in frames:
            bb, tt, device = emb.shape[0], emb.shape[1], emb.device
            if self.bypass_quantizer:
                code_embs, indices, sub_quants = emb, torch.zeros(bb, tt, dtype=torch.long, device=device), torch.zeros_like(emb, device=device)
            else:
                quant_in = emb
                quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
                code_embs = quant_out
            codes.append((code_embs, scale if use_scale else None))
            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval

    def inference_encoding(
            self,
            speech: torch.Tensor,
            need_recon: bool = False,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            speech (torch.Tensor): input speech
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = []
        code_idxs = []
        all_sub_quants = []
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        frames = self._encode(speech)
        for emb, scale in frames:
            quant_in = emb
            quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
            code_embs = quant_out
            codes.append((code_embs, scale if use_scale else None))
            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval

    def inference_decoding(
            self,
            token_idx: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            token_idx (torch.Tensor): input token indices, B x T x n_q
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = []
        token_idx = token_idx.permute(2, 0, 1).unsqueeze(0)
        for tokens in token_idx:
            code_embs = self.quantizer.decode(tokens)
            codes.append((code_embs.transpose(1, 2), None))
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)
        retval = dict(
            recon_speech=recon_speech,
            code_indices=None,
            code_embeddings=codes,
            sub_quants=None
        )
        return retval

    def inference_decoding_emb(
            self,
            token_idx: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            token_idx (torch.Tensor): input code embeddings, B x T x Dim
            need_recon (bool): whether to return recon speech
            bit_width (int): The excepted bandwidth
            use_scale (bool): whether to use scale

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (T_wav,).
                * code_indices (Tensor): quantized code indices (L)
                * code_embeddings (Tensor): quantized code embeddings (L, d).

        """
        codes = [(token_idx, None)]
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)
        retval = dict(
            recon_speech=recon_speech,
            code_indices=None,
            code_embeddings=codes,
            sub_quants=None
        )
        return retval

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
