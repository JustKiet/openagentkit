from speechbrain.inference.VAD import VAD
from openagentkit._types import NamedBytesIO

class VADEngine:
    def __init__(self):
        self.vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    def apply_vad(self, 
                  audio_bytes: NamedBytesIO,
                  activation_th: float = 0.5,
                  deactivation_th: float = 0.25) -> bytes:
        """
        Apply Voice Activity Detection (VAD) to the audio bytes.
        
        :param audio_bytes: NamedBytesIO object containing audio data
        :param activation_th: Activation threshold for VAD (default: 0.5)
        :param deactivation_th: Deactivation threshold for VAD (default: 0.25)
        :return: Bytes object containing the VAD-processed audio
        """
        prob_chunks = self.vad.get_speech_prob_file(audio_bytes)
        prob_threshold = self.vad.apply_threshold(
            vad_prob=prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th
        )

        boundaries = self.vad.get_boundaries(
            vad_prob=prob_threshold,
        )

        boundaries = self.vad.merge_close_segments(boundaries, close_th=0.250)

        boundaries = self.vad.remove_short_segments(boundaries, len_th=0.250)

        return boundaries.numpy().tobytes()

