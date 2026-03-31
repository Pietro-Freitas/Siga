"""
speech_engine.py — Motor de Reconhecimento de Voz (STT)
========================================================
Escuta o microfone continuamente em thread dedicada e publica
comandos reconhecidos em uma fila para o Orquestrador processar.

Isolamento total: falhas de áudio não afetam a câmera ou o TTS.
"""

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import sounddevice as sd
import vosk

from config import MODEL, STT

logger = logging.getLogger(__name__)


class SpeechEngine:
    """
    Motor de reconhecimento de voz offline baseado no Vosk.

    Roda em thread daemon e deposita strings de comandos reconhecidos
    em `command_queue`, que o Orquestrador consome.

    Parameters
    ----------
    command_queue:
        Fila onde os comandos reconhecidos são publicados.
    model_path:
        Caminho para o diretório do modelo Vosk.

    Exemplo::

        cmds: queue.Queue[str] = queue.Queue()
        stt = SpeechEngine(cmds)
        stt.start()
        cmd = cmds.get()   # bloqueia até chegar um comando
    """

    def __init__(
        self,
        command_queue: "queue.Queue[str]",
        model_path: Path = MODEL.vosk,
    ) -> None:
        self._cmd_queue  = command_queue
        self._model_path = model_path
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia thread de escuta de microfone. Idempotente."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            name="SpeechEngine-STT",
            daemon=True,
        )
        self._thread.start()
        logger.info("SpeechEngine iniciada.")

    def stop(self) -> None:
        """Solicita encerramento da thread de escuta."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("SpeechEngine encerrada.")

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _load_model(self) -> Optional[vosk.Model]:
        """Carrega o modelo Vosk com tratamento de erros."""
        if not self._model_path.exists():
            logger.error("Modelo Vosk não encontrado: %s", self._model_path)
            return None
        try:
            model = vosk.Model(str(self._model_path))
            logger.info("Modelo Vosk carregado: %s", self._model_path.name)
            return model
        except Exception:
            logger.exception("Falha ao carregar modelo Vosk.")
            return None

    def _worker(self) -> None:
        """Loop principal da thread STT."""
        model = self._load_model()
        if model is None:
            logger.error("STT desabilitado por falha no carregamento do modelo.")
            return

        rec = vosk.KaldiRecognizer(model, STT.samplerate)

        def _audio_callback(indata, frames, time_info, status):
            """Callback do sounddevice — chamado pela lib de áudio."""
            if status:
                logger.warning("Status do stream de áudio: %s", status)
            if rec.AcceptWaveform(bytes(indata)):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    logger.info("Comando reconhecido: '%s'", text)
                    self._cmd_queue.put(text)

        try:
            with sd.RawInputStream(
                samplerate=STT.samplerate,
                blocksize=STT.blocksize,
                dtype=STT.dtype,
                channels=STT.channels,
                callback=_audio_callback,
            ):
                logger.info("Microfone aberto — aguardando comandos...")
                while self._running:
                    # 500ms de sleep reduz wakeups desnecessários
                    # (o callback de áudio roda independentemente)
                    sd.sleep(500)
        except Exception:
            logger.exception("Erro fatal no stream de áudio STT.")
        finally:
            logger.debug("Worker STT encerrado.")
