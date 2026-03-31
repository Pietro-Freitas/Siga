"""
voice_interface.py — Motor de Síntese de Voz (TTS) Não-Bloqueante
==================================================================
Executa em uma thread dedicada, consumindo mensagens de uma fila.
O loop principal da câmera NUNCA é bloqueado por síntese de fala.

Padrão de projeto: Producer-Consumer com fila bounded.
Anti-spam: mensagens idênticas recentes são descartadas automaticamente.
Limpeza periódica do cache anti-spam para evitar memory leak.
"""

import logging
import queue
import threading
import time
from typing import Optional

import pyttsx3

from config import VOICE

logger = logging.getLogger(__name__)


class VoiceInterface:
    """
    Interface de síntese de voz thread-safe para o SIGA.

    Utiliza uma fila interna (bounded) para desacoplar completamente
    a produção de mensagens (detecções, comandos) da execução de TTS,
    que é inerentemente lenta e bloqueante.

    Controle de spam integrado: a mesma mensagem não é repetida antes
    de `spam_interval` segundos, evitando sobrecarga auditiva ao usuário.

    Exemplo de uso::

        voz = VoiceInterface()
        voz.start()
        voz.falar("Obstáculo à frente")   # não-bloqueante
        voz.falar("Obstáculo à frente")   # ignorado (anti-spam)
        voz.stop()
    """

    def __init__(
        self,
        spam_interval: float = VOICE.spam_interval,
        queue_maxsize: int = VOICE.tts_queue_maxsize,
    ) -> None:
        self._spam_interval  = spam_interval
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=queue_maxsize)
        self._last_spoken:   dict[str, float]  = {}   # texto → timestamp
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._engine: Optional[pyttsx3.Engine] = None
        self._last_cleanup = time.monotonic()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia a thread de TTS. Idempotente."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            name="VoiceInterface-TTS",
            daemon=True,
        )
        self._thread.start()
        logger.info("VoiceInterface iniciada.")

    def stop(self, timeout: float = 3.0) -> None:
        """Solicita encerramento gracioso da thread de TTS."""
        if not self._running:
            return
        self._running = False
        self._queue.put(None)  # sentinela para desbloquear o worker
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("VoiceInterface encerrada.")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def falar(self, texto: str, prioridade: bool = False) -> bool:
        """
        Enfileira `texto` para síntese de voz de forma não-bloqueante.

        Parameters
        ----------
        texto:      Mensagem a ser falada.
        prioridade: Se True, ignora o controle de spam (ex: alertas críticos).

        Returns
        -------
        True se a mensagem foi enfileirada, False se descartada (spam/fila cheia).
        """
        if not texto or not texto.strip():
            return False

        texto = texto.strip()

        if not prioridade and self._is_spam(texto):
            logger.debug("Anti-spam: descartando '%s'", texto)
            return False

        try:
            self._queue.put_nowait(texto)
            with self._lock:
                self._last_spoken[texto] = time.monotonic()
            logger.debug("TTS enfileirado: '%s'", texto)
            return True
        except queue.Full:
            logger.warning("Fila TTS cheia — mensagem descartada: '%s'", texto)
            return False

    def falar_agora(self, texto: str) -> None:
        """
        Fala imediatamente (bloqueante). Reservado para inicialização/shutdown.
        NÃO use dentro do loop de câmera.

        Se a engine TTS não estiver inicializada (ex: chamado antes do start()),
        cria uma engine temporária para não falhar silenciosamente.
        """
        if self._engine:
            self._sintetizar(texto)
        else:
            # Engine ainda não criada — cria uma temporária
            try:
                engine = pyttsx3.init()
                engine.setProperty("volume", VOICE.volume)
                engine.setProperty("rate", VOICE.rate)
                engine.say(texto)
                engine.runAndWait()
            except Exception:
                logger.exception("Erro ao sintetizar (engine temporária): '%s'", texto)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _is_spam(self, texto: str) -> bool:
        with self._lock:
            last = self._last_spoken.get(texto)
        return last is not None and (time.monotonic() - last) < self._spam_interval

    def _cleanup_spam_cache(self) -> None:
        """Remove entradas expiradas do cache anti-spam para evitar memory leak."""
        now = time.monotonic()
        if now - self._last_cleanup < VOICE.spam_cleanup_interval:
            return
        self._last_cleanup = now
        with self._lock:
            expired = [
                k for k, v in self._last_spoken.items()
                if (now - v) > self._spam_interval * 3
            ]
            for k in expired:
                del self._last_spoken[k]
            if expired:
                logger.debug("Spam cache: %d entradas limpas.", len(expired))

    def _worker(self) -> None:
        """Loop da thread de TTS: inicializa engine e consome a fila."""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("volume", VOICE.volume)
            self._engine.setProperty("rate",   VOICE.rate)
            logger.debug("Engine TTS inicializada na thread worker.")
        except Exception:
            logger.exception("Falha ao inicializar engine TTS. Thread encerrando.")
            self._running = False
            return

        while self._running:
            try:
                texto = self._queue.get(timeout=0.3)
                if texto is None:           # sentinela de encerramento
                    break
                self._sintetizar(texto)
                self._queue.task_done()
            except queue.Empty:
                pass
            except Exception:
                logger.exception("Erro inesperado no worker TTS.")

            # Limpeza periódica do cache
            self._cleanup_spam_cache()

        logger.debug("Worker TTS encerrado.")

    def _sintetizar(self, texto: str) -> None:
        """Executa síntese de voz de forma segura."""
        try:
            if self._engine:
                self._engine.say(texto)
                self._engine.runAndWait()
        except Exception:
            logger.exception("Erro ao sintetizar: '%s'", texto)
