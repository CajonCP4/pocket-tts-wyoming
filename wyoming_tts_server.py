#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import re
from functools import partial
from typing import Optional

import numpy
from pocket_tts import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.utils.utils import PREDEFINED_VOICES
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = int(os.environ.get("WYOMING_PORT", "10201"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "alba")
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", DEFAULT_VARIANT)

# Tuning Parameter
PREFIX_MIN_DURATION = 0.1
PREFIX_MAX_DURATION = 0.8 # Kürzerer Suchzeitraum für schnellere Reaktion
PREFIX_SILENCE_GAP = 0.06

_VOICE_STATES: dict[str, dict] = {}
_VOICE_LOCK = asyncio.Lock()

class PocketTTSEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, cli_args: argparse.Namespace, tts_model: TTSModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.tts_model = tts_model
        self.is_streaming: bool = False
        self._text_buffer: str = ""
        self._voice_info: Optional[any] = None
        self._audio_started: bool = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            if Synthesize.is_type(event.type):
                synthesize = Synthesize.from_event(event)
                await self.write_event(SynthesizeStart(voice=synthesize.voice).event())
                await self._handle_synthesize(synthesize.text, synthesize.voice, is_last=True)
                await self.write_event(SynthesizeStopped().event())
                return True

            if SynthesizeStart.is_type(event.type):
                start_event = SynthesizeStart.from_event(event)
                _LOGGER.info("LLM Text-Stream gestartet")
                # Wir senden SynthesizeStart sofort zurück (HA Timer an)
                await self.write_event(start_event.event())
                self.is_streaming = True
                self._text_buffer = ""
                self._voice_info = start_event.voice
                self._audio_started = False
                return True

            if SynthesizeChunk.is_type(event.type):
                chunk = SynthesizeChunk.from_event(event)
                self._text_buffer += chunk.text
                # Wir splitten bei Satzzeichen
                if any(c in self._text_buffer for c in ".!?\n"):
                    parts = re.split(r'([.!?\n]+)', self._text_buffer)
                    if len(parts) > 2:
                        to_speak = "".join(parts[:-1]).strip()
                        self._text_buffer = parts[-1]
                        if to_speak:
                            _LOGGER.info("Satz erkannt: '%s'", to_speak)
                            await self._handle_synthesize(to_speak, self._voice_info, is_last=False)
                return True

            if SynthesizeStop.is_type(event.type):
                if self._text_buffer.strip():
                    await self._handle_synthesize(self._text_buffer.strip(), self._voice_info, is_last=True)
                else:
                    # Falls kein Text mehr da war, müssen wir trotzdem den Audiostream schließen
                    if self._audio_started:
                        await self.write_event(AudioStop().event())

                await self.write_event(SynthesizeStopped().event())
                _LOGGER.info("LLM Text-Stream beendet.")
                self.is_streaming = False
                self._audio_started = False
                return True

            return True
        except Exception as err:
            _LOGGER.error("Fehler: %s", err, exc_info=True)
            return False

    async def _handle_synthesize(self, text: str, voice_obj: Optional[any], is_last: bool) -> bool:
        text_to_speak = text.strip()
        if not text_to_speak: return True

        # Sacrificial Prefix
        processed_text = "... " + text_to_speak
        
        voice_name = voice_obj.name if voice_obj else self.cli_args.voice
        if voice_name and voice_name.startswith("pocket-tts-"):
            voice_name = voice_name.replace("pocket-tts-", "", 1)
        if voice_name not in PREDEFINED_VOICES:
            voice_name = self.cli_args.voice

        async with _VOICE_LOCK:
            if voice_name not in _VOICE_STATES:
                method = getattr(self.tts_model, 'get_state_for_audio_prompt', 
                                getattr(self.tts_model, 'get_state_for_voice', None))
                _VOICE_STATES[voice_name] = method(voice_name)
            
            voice_state = _VOICE_STATES[voice_name]

            try:
                sample_rate = self.tts_model.sample_rate
                
                # Sende AudioStart nur beim ERSTEN Satz einer Session
                if not self._audio_started:
                    await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
                    self._audio_started = True

                audio_stream = self.tts_model.generate_audio_stream(
                    model_state=voice_state, text_to_generate=processed_text, copy_state=True
                )

                prefix_buffer = numpy.array([], dtype=numpy.float32)
                prefix_processed = False
                max_prefix_samples = int(sample_rate * PREFIX_MAX_DURATION)
                
                for chunk_tensor in audio_stream:
                    chunk_data = chunk_tensor.detach().cpu().numpy().flatten()
                    if not prefix_processed:
                        prefix_buffer = numpy.concatenate([prefix_buffer, chunk_data])
                        if len(prefix_buffer) >= max_prefix_samples:
                            valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                            await self._send_audio_chunks(valid_audio, sample_rate)
                            prefix_processed = True
                            prefix_buffer = None
                    else:
                        await self._send_audio_chunks(chunk_data, sample_rate)
                        await asyncio.sleep(0)

                if not prefix_processed and len(prefix_buffer) > 0:
                     valid_audio = self._trim_prefix(prefix_buffer, sample_rate)
                     await self._send_audio_chunks(valid_audio, sample_rate)

                # WICHTIG: AudioStop NUR senden, wenn dies wirklich das Ende des LLM Streams ist
                if is_last:
                    await self.write_event(AudioStop().event())
                
            except Exception as e:
                _LOGGER.error("Generierungsfehler: %s", e)
                return False
        return True

    def _trim_prefix(self, audio_data, sample_rate):
        if len(audio_data) == 0: return audio_data
        
        # Sicherheits-Check: Wenn das Audio extrem kurz ist, trimme fast gar nichts
        if len(audio_data) < int(sample_rate * 0.5):
            return audio_data[int(sample_rate * 0.05):]

        threshold = numpy.abs(audio_data).max() * 0.01
        min_prefix = int(sample_rate * PREFIX_MIN_DURATION)
        max_prefix = int(sample_rate * PREFIX_MAX_DURATION)
        silence_gap = int(sample_rate * PREFIX_SILENCE_GAP)
        
        prefix_end = 0
        search_end = min(len(audio_data), max_prefix)
        
        is_silent = numpy.abs(audio_data[:search_end]) < threshold
        i = min_prefix
        while i < search_end:
            if is_silent[i]:
                start = i
                while i < search_end and is_silent[i]: i += 1
                if (i - start) >= silence_gap:
                    prefix_end = i
                    break
            i += 1
            
        # Wenn kein Gap gefunden wurde, trimme nur das absolute Minimum (0.05s)
        # statt dem User den ganzen Satz wegzuschneiden
        if prefix_end == 0:
            return audio_data[int(sample_rate * 0.05):]
            
        return audio_data[prefix_end:]

    async def _send_audio_chunks(self, float_audio, rate):
        if len(float_audio) == 0: return
        _LOGGER.info("DEBUG STREAM: Sende %d Samples", len(float_audio))
        audio_int16 = (float_audio.clip(-1.0, 1.0) * 32767).astype("int16")
        audio_bytes = audio_int16.tobytes()
        chunk_size = 2048 
        for i in range(0, len(audio_bytes), chunk_size):
            await self.write_event(AudioChunk(rate=rate, width=2, channels=1, audio=audio_bytes[i:i+chunk_size]).event())

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10201)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--variant", default=MODEL_VARIANT)
    parser.add_argument("--uri", default=None)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        tts_model = TTSModel.load_model(args.variant)
    except:
        tts_model = TTSModel.load_model(model_variant_id=args.variant)

    voices = [TtsVoice(name=v, description=f"Pocket-TTS {v}", attribution=Attribution(name="Kyutai", url="https://kyutai.org"),
              installed=True, languages=["en"], version="1.0.1", speakers=None) for v in PREDEFINED_VOICES]

    wyoming_info = Info(tts=[TtsProgram(name="pocket-tts", description="Fast Streaming Pocket-TTS",
        attribution=Attribution(name="Kyutai", url="https://kyutai.org"), installed=True, voices=voices, 
        version="1.0.1", supports_synthesize_streaming=True)])

    if args.uri is None: args.uri = f"tcp://{args.host}:{args.port}"
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Server bereit auf %s", args.uri)
    await server.run(partial(PocketTTSEventHandler, wyoming_info, args, tts_model))

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
