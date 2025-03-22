import io
import wave
from loguru import logger
from typing import Literal
import struct

class AudioUtility:
    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> Literal["wav", "webm", "mp3", "ogg", "flac", "aac", "aiff", "mpeg", "mpga", "m4a", "pcm", "unknown"]:
        """
        Detect the format of audio data based on file signatures.
        
        :param audio_bytes: Raw audio data bytes
        :return: String indicating the detected format
        """
        if len(audio_bytes) < 12:
            logger.warning("Audio data too short to determine format")
            return "unknown"
            
        # Check for WAV (RIFF header)
        if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[0:12]:
            logger.info("Detected WAV format")
            return "wav"
            
        # Check for WebM
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
            logger.info("Detected WebM format")
            return "webm"
            
        # Check for MP3
        if audio_bytes.startswith(b'\xFF\xFB') or audio_bytes.startswith(b'\x49\x44\x33'):
            logger.info("Detected MP3 format")
            return "mp3"
            
        # Check for Ogg Vorbis
        if audio_bytes.startswith(b'OggS'):
            logger.info("Detected Ogg format")
            return "ogg"
            
        # Check for FLAC
        if audio_bytes.startswith(b'fLaC'):
            logger.info("Detected FLAC format")
            return "flac"
            
        # Check for AAC
        if audio_bytes.startswith(b'\xFF\xF1') or audio_bytes.startswith(b'\xFF\xF9'):
            logger.info("Detected AAC format")
            return "aac"
            
        # Check for AIFF
        if audio_bytes.startswith(b'FORM') and b'AIFF' in audio_bytes[0:12]:
            logger.info("Detected AIFF format")
            return "aiff"
            
        # Check for M4A (MPEG-4 Audio)
        if audio_bytes.startswith(b'\x00\x00\x00\x20\x66\x74\x79\x70\x4D\x34\x41') or \
           audio_bytes.startswith(b'\x00\x00\x00\x18\x66\x74\x79\x70\x6D\x70\x34\x32'):
            logger.info("Detected M4A format")
            return "m4a"
            
        # Check for MPEG
        if audio_bytes.startswith(b'\x00\x00\x01\xBA') or audio_bytes.startswith(b'\x00\x00\x01\xB3'):
            logger.info("Detected MPEG format")
            return "mpeg"
            
        # Check for MPGA (MPEG-1 Layer 3)
        if len(audio_bytes) > 2 and (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
            logger.info("Detected MPGA format")
            return "mpga"
        
        # If no known signatures match, try to determine if it's raw PCM
        try:
            # Check if data looks like 16-bit PCM (reasonable amplitude values)
            if len(audio_bytes) >= 1000:  # Need a reasonable amount of data
                # Sample a few values
                samples = []
                for i in range(0, min(1000, len(audio_bytes) - 1), 2):
                    if i + 1 < len(audio_bytes):
                        sample = struct.unpack('<h', audio_bytes[i:i+2])[0]
                        samples.append(abs(sample))
                
                # Check if values are within reasonable range for audio
                avg = sum(samples) / len(samples)
                if 0 < avg < 32768:  # 16-bit audio range
                    logger.info("Detected likely raw PCM data")
                    return "pcm"
        except Exception as e:
            logger.error(f"Error checking for PCM: {e}")
        
        logger.warning("Unknown audio format")
        return "unknown"
    
    @staticmethod
    def validate_wav(wav_bytes: bytes) -> bool:
        """
        Validate if the given BytesIO object contains a valid WAV file.
        """
        try:
            with io.BytesIO(wav_bytes) as wav_file:
                with wave.open(wav_file, 'rb') as wf:
                    num_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    frame_rate = wf.getframerate()
                    num_frames = wf.getnframes()

                    logger.info(f"Valid WAV File: {num_channels} channels, {sample_width*8}-bit, {frame_rate}Hz, {num_frames} frames")
                    return True
        except wave.Error as e:
            logger.error(f"Invalid WAV File: {e}")
            return False
    
    @staticmethod
    def raw_bytes_to_wav(raw_audio_bytes, 
                         sample_rate=16000,  # Whisper prefers 16kHz
                         num_channels=1,     # Mono is better for speech recognition
                         sample_width=2) -> io.BytesIO:  # 16-bit audio
        """
        Convert raw PCM audio bytes into a WAV file-like object.
        
        :param raw_audio_bytes: Raw PCM audio data (bytes)
        :param sample_rate: Sample rate in Hz
        :param num_channels: Number of audio channels (1=mono, 2=stereo)
        :param sample_width: Sample width in bytes (2 for 16-bit audio)
        :return: A BytesIO object containing the WAV file
        """
        # Log the size of incoming data
        logger.info(f"Converting {len(raw_audio_bytes)} bytes of raw audio data to WAV")
        
        # Create a new WAV file in memory
        wav_file = io.BytesIO()
        
        try:
            with wave.open(wav_file, 'wb') as wf:
                wf.setnchannels(num_channels)      # Mono for speech recognition
                wf.setsampwidth(sample_width)      # 2 bytes = 16-bit PCM
                wf.setframerate(sample_rate)       # 16kHz for Whisper
                wf.writeframes(raw_audio_bytes)    # Write raw PCM audio data

            wav_file.seek(0)  # Move back to start for reading
            
            # Verify the WAV file is valid
            wav_file_copy = io.BytesIO(wav_file.getvalue())
            with wave.open(wav_file_copy, 'rb') as wf:
                logger.info(f"Created WAV: {wf.getnchannels()} channels, {wf.getsampwidth()*8}-bit, {wf.getframerate()}Hz, {wf.getnframes()} frames")
            
            return wav_file
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            # Return an empty WAV file with correct headers
            empty_wav = io.BytesIO()
            with wave.open(empty_wav, 'wb') as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(b'')  # Empty audio
            empty_wav.seek(0)
            return empty_wav