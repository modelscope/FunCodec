import torch
from typing import Tuple, List, Dict, Any
from funcodec.models.discriminator.hifigan import HiFiGANMultiPeriodDiscriminator
from funcodec.models.discriminator.hifigan import HiFiGANMultiScaleDiscriminator
from funcodec.models.discriminator.hifigan import HiFiGANMultiScaleMultiPeriodDiscriminator
from funcodec.models.discriminator.hifigan import HiFiGANPeriodDiscriminator
from funcodec.models.discriminator.hifigan import HiFiGANScaleDiscriminator
from funcodec.models.discriminator.sound_stream import ComplexSTFTDiscriminator
from funcodec.models.discriminator.sound_stream import MultiScaleDiscriminator
from funcodec.models.discriminator.encodec_disc import MultiScaleSTFTDiscriminator


class MultipleDiscriminator(torch.nn.Module):
    def __init__(
            self,
            input_size=1,
            disc_conf_list: List[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.support_disc_choices = dict(
            hifigan_period_discriminator=HiFiGANPeriodDiscriminator,
            hifigan_scale_discriminator=HiFiGANScaleDiscriminator,
            hifigan_multi_period_discriminator=HiFiGANMultiPeriodDiscriminator,
            hifigan_multi_scale_discriminator=HiFiGANMultiScaleDiscriminator,
            hifigan_multi_scale_multi_period_discriminator=HiFiGANMultiScaleMultiPeriodDiscriminator,
            soundstream_complex_stft_discriminator=ComplexSTFTDiscriminator,
            soundstream_multi_scale_discriminator=MultiScaleDiscriminator,
            encodec_multi_scale_stft_discriminator=MultiScaleSTFTDiscriminator,
        )
        self.discriminators = torch.nn.ModuleList([])
        for args in disc_conf_list:
            assert "name" in args, "disc_conf must have `name` attr to specific disc type."
            disc_type = args.pop("name")
            assert disc_type in self.support_disc_choices, \
                "Unsupported discriminator type, only support {}".format(
                    ",".join(self.support_disc_choices.keys())
                )

            disc_class = self.support_disc_choices[disc_type]
            one_disc = disc_class(in_channels=input_size, **args)
            self.discriminators.append(one_disc)
            # add back to the args for dump config.yaml
            args["name"] = disc_type

    def forward(self, x, return_intermediates=True):
        retval = []
        for disc in self.discriminators:
            out = disc(x, return_intermediates=return_intermediates)
            if isinstance(out, tuple):
                retval.append(out)
            elif isinstance(out, list):
                retval.extend(out)
            else:
                raise TypeError("The return value of discriminator must be tuple or list[tuple]")

        return retval
