import io
import wave
import logging
from typing import Literal, TypeAlias
import struct

logger = logging.getLogger(__name__)

AudioFormat: TypeAlias = Literal[
    "wav", "webm", "mp3", "ogg", "flac", "aac", "aiff",
    "mpeg", "mpga", "m4a", "pcm", "unknown"
]

class AudioUtility:
    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> AudioFormat:
        """
        Detect the format of audio data based on file signatures.
        
        :param audio_bytes: Raw audio data bytes
        :return: String indicating the detected format
        """
        if len(audio_bytes) < 12:
            return "unknown"
        
        # Define signature patterns for quick lookup
        format_signatures = {
            "wav": [(b'RIFF', 0, 4), (b'WAVE', 8, 12)],
            "webm": [(b'\x1A\x45\xDF\xA3', 0, 4)],
            "mp3": [(b'\xFF\xFB', 0, 2), (b'\xFF\xF3', 0, 2), (b'\xFF\xF2', 0, 2), (b'\x49\x44\x33', 0, 3)],
            "ogg": [(b'OggS', 0, 4)],
            "flac": [(b'fLaC', 0, 4)],
            "aac": [(b'\xFF\xF1', 0, 2), (b'\xFF\xF9', 0, 2)],
            "aiff": [(b'FORM', 0, 4), (b'AIFF', 8, 12)],
            "mpeg": [(b'\x00\x00\x01\xBA', 0, 4), (b'\x00\x00\x01\xB3', 0, 4)],
        }
        
        # Check common formats using signature lookup
        for fmt, signatures in format_signatures.items():
            for sig, start, end in signatures:
                if start + len(sig) <= len(audio_bytes) and audio_bytes[start:end].startswith(sig):
                    if fmt == "wav" and not any(s[0] == b'WAVE' for s in signatures) or fmt != "wav":
                        return 'wav'
        
        # Additional checks for formats that need special handling
        # WebM - check for additional signatures if primary one didn't match
        if (b'\x42\x82\x84webm' in audio_bytes[:50] or 
            b'\x1A\x45\xDF\xA3\x01\x00\x00\x00' in audio_bytes[:50] or
            b'audio/webm' in audio_bytes[:1000]):
            return "webm"
            
        # M4A/AAC detection
        if b'ftypM4A' in audio_bytes[:20]:
            return "m4a"
            
        # MPGA check
        if len(audio_bytes) > 2 and (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
            return "mpga"
        
        # WAV metadata in browser recording
        if b'audio/wav' in audio_bytes[:1000] or b'audio/x-wav' in audio_bytes[:1000]:
            return "wav"
            
        # PCM detection - only do this if other formats don't match
        if len(audio_bytes) >= 1000:
            try:
                # Sample fewer values for speed
                samples = [abs(struct.unpack('<h', audio_bytes[i:i+2])[0]) 
                          for i in range(0, 1000, 40) if i+1 < len(audio_bytes)]
                
                if samples and 0 < sum(samples)/len(samples) < 32768:
                    return "pcm"
            except:
                pass
        
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
    def raw_bytes_to_wav(raw_audio_bytes: bytes, 
                         sample_rate: int = 16000,  # Whisper prefers 16kHz
                         num_channels: int = 1,     # Mono is better for speech recognition
                         sample_width: int = 2) -> io.BytesIO:  # 16-bit audio
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
        
        # Check if input might already be a WAV file
        if len(raw_audio_bytes) > 44 and raw_audio_bytes.startswith(b'RIFF') and b'WAVE' in raw_audio_bytes[:12]:
            logger.info("Input appears to be already in WAV format, returning as is")
            return io.BytesIO(raw_audio_bytes)
            
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
            
    @staticmethod
    def convert_audio_format(audio_bytes: bytes, source_format: str, target_format: str = "wav") -> bytes | None:
        """
        Convert audio from one format to another using FFmpeg.
        
        :param audio_bytes: Input audio data in bytes
        :param source_format: Source format (e.g., 'webm', 'mp3')
        :param target_format: Target format (e.g., 'wav', 'mp3')
        :return: Converted audio data in bytes
        """
        try:
            import tempfile
            import subprocess
            import os
            
            # Create temp files for input and output
            with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as in_file:
                in_file.write(audio_bytes)
                in_path = in_file.name
                
            out_path = in_path.replace(f'.{source_format}', f'.{target_format}')
            
            # Run FFmpeg conversion
            logger.info(f"Converting {source_format} to {target_format} using FFmpeg")
            command = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-i', in_path,  # Input file
                '-ar', '16000',  # Output sample rate (16kHz for Whisper)
                '-ac', '1',      # Mono audio
                '-c:a', 'pcm_s16le' if target_format == 'wav' else 'libmp3lame',  # Codec
                out_path  # Output file
            ]
            
            # Execute ffmpeg and capture output
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {process.stderr.decode()}")
                return None
                
            # Read the converted file
            with open(out_path, 'rb') as out_file:
                converted_data = out_file.read()
                
            # Clean up temp files
            os.unlink(in_path)
            os.unlink(out_path)
            
            logger.info(f"Successfully converted {len(audio_bytes)} bytes from {source_format} to {len(converted_data)} bytes of {target_format}")
            return converted_data
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return None