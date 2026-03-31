"""
vision_engine.py — Motor de Visão com Recuperação Automática de Câmera
=======================================================================
Roda em thread dedicada e garante que o loop de câmera NUNCA pare,
mesmo diante de falhas de hardware ou desconexões momentâneas.

Publica resultados de inferência via callback para o Orquestrador,
que decide o que falar e exibir.

Separação de responsabilidades:
  • VisionEngine  → captura + inferência + recuperação de câmera
  • ModelManager  → toda a lógica de ML
  • SigaOrchestrator → integra tudo e comanda o TTS

Otimizações de performance:
  • Frame skipping: se a inferência demorar mais que o intervalo
    entre frames, o buffer da câmera é drenado para descartar
    frames velhos e processar sempre o mais recente.
  • Buffer size = 1: câmera configurada para manter no máximo
    1 frame no buffer interno, eliminando latência acumulada.
  • FPS throttling: garante não processar mais frames que o alvo.
"""

import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from config import CAMERA, DISPLAY
from model_manager import InferenceResult, ModelManager

logger = logging.getLogger(__name__)


class VisionEngine:
    """
    Motor de visão do SIGA: captura contínua + inferência orquestrada.

    A câmera é aberta com tentativas de reconexão automática.
    Cada frame processado gera um `InferenceResult` entregue ao
    callback `on_result`, que roda *na mesma thread* (evita cópias).

    Parameters
    ----------
    model_manager:
        Instância pronta do ModelManager.
    on_result:
        Callable invocado após cada inferência com (frame, InferenceResult).
        Deve ser rápido — não bloqueie aqui.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        on_result: Callable[[np.ndarray, InferenceResult], None],
    ) -> None:
        self._mm        = model_manager
        self._on_result = on_result
        self._thread: Optional[threading.Thread] = None
        self._running   = False
        self._frame_id  = 0
        self._cap: Optional[cv2.VideoCapture] = None

        # Controle de FPS real
        self._frame_interval = 1.0 / max(CAMERA.fps_target, 1)

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia a thread de visão. Idempotente."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="VisionEngine-Camera",
            daemon=True,
        )
        self._thread.start()
        logger.info("VisionEngine iniciada.")

    def stop(self) -> None:
        """Encerra o loop de câmera de forma graciosa."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._release_camera()
        logger.info("VisionEngine encerrada.")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """
        Loop de captura e inferência. Nunca retorna enquanto `_running=True`.
        Reconecta automaticamente a câmera em caso de falha.

        Throttling de FPS:
          Se a inferência terminar antes do intervalo alvo, o loop espera
          o tempo restante. Se demorar mais, drena frames acumulados do
          buffer da câmera para processar sempre o frame mais recente.
        """
        attempts = 0

        while self._running:
            # --- Tenta abrir/reconectar câmera ---
            if not self._ensure_camera_open():
                attempts += 1
                if attempts > CAMERA.max_reconnect_attempts:
                    logger.critical(
                        "Câmera falhou %d vezes consecutivas. "
                        "Verifique o hardware.",
                        attempts,
                    )
                    # Pausa longa antes de tentar de novo
                    time.sleep(CAMERA.reconnect_delay * 5)
                    attempts = 0
                continue

            attempts = 0   # reset em caso de sucesso de abertura

            frame_start = time.perf_counter()

            # --- Captura ---
            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.warning("Frame inválido — possível desconexão da câmera.")
                self._release_camera()
                time.sleep(CAMERA.reconnect_delay)
                continue

            self._frame_id += 1

            # --- Inferência ---
            try:
                result = self._mm.infer(frame, self._frame_id)
            except Exception:
                logger.exception("Erro de inferência no frame %d.", self._frame_id)
                result = InferenceResult(frame_id=self._frame_id)

            # --- Callback para o Orquestrador ---
            try:
                self._on_result(frame, result)
            except Exception:
                logger.exception("Erro no callback on_result.")

            # --- Throttling de FPS ---
            elapsed = time.perf_counter() - frame_start
            remaining = self._frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -self._frame_interval:
                # Inferência demorou mais que 2× o intervalo.
                # Drena frames acumulados no buffer para não processar dados velhos.
                self._drain_camera_buffer()

        logger.debug("Loop de câmera encerrado.")

    def _drain_camera_buffer(self) -> None:
        """
        Descarta frames antigos acumulados no buffer da câmera.
        Garante que o próximo frame lido seja o mais recente disponível.
        """
        if self._cap is None:
            return
        for _ in range(3):  # máximo de 3 frames descartados
            self._cap.grab()

    # ------------------------------------------------------------------
    # Câmera helpers
    # ------------------------------------------------------------------

    def _ensure_camera_open(self) -> bool:
        """Garante que a câmera está aberta e responsiva."""
        if self._cap is not None and self._cap.isOpened():
            return True

        logger.info(
            "Tentando abrir câmera (device=%d)...", CAMERA.device
        )
        try:
            cap = cv2.VideoCapture(CAMERA.device)
            if not cap.isOpened():
                logger.warning("cv2.VideoCapture falhou — tentando novamente em %.1fs.",
                               CAMERA.reconnect_delay)
                time.sleep(CAMERA.reconnect_delay)
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
            cap.set(cv2.CAP_PROP_FPS,          CAMERA.fps_target)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   CAMERA.buffer_size)
            self._cap = cap
            logger.info(
                "Câmera aberta — %dx%d @%dfps (buffer=%d)",
                CAMERA.width, CAMERA.height, CAMERA.fps_target,
                CAMERA.buffer_size,
            )
            return True
        except Exception:
            logger.exception("Exceção ao abrir câmera.")
            time.sleep(CAMERA.reconnect_delay)
            return False

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


# ---------------------------------------------------------------------------
# Renderizador de frame (separado do motor para manter SRP)
# ---------------------------------------------------------------------------

class FrameRenderer:
    """
    Responsável exclusivamente por desenhar as detecções sobre o frame
    e exibir a janela de debug.

    Não contém lógica de negócio — apenas visualização.

    NOTA PI 2/3: Em deploy headless, `DISPLAY.show_window=False` faz com que
    este renderer nunca seja chamado, economizando CPU.
    Consulte docs/pi2_deployment.md ou docs/pi3_deployment.md.
    """

    @staticmethod
    def render(
        frame: np.ndarray,
        result: InferenceResult,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Desenha bounding boxes e labels no frame.

        Parameters
        ----------
        frame:  Frame BGR original.
        result: Resultado de inferência com detecções.
        copy:   Se True, trabalha em cópia (seguro). Se False, modifica
                o frame original in-place (mais rápido, use com cautela).

        Returns
        -------
        Frame anotado.
        """
        annotated = frame.copy() if copy else frame

        for det in result.detections:
            cv2.rectangle(
                annotated,
                (det.x1, det.y1),
                (det.x2, det.y2),
                det.color,
                DISPLAY.thickness,
            )
            label = f"{det.name} {det.conf:.0%}"
            cv2.putText(
                annotated,
                label,
                (det.x1, max(det.y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                DISPLAY.font_scale,
                det.color,
                DISPLAY.thickness,
            )

        # Info de performance no canto
        fps_text = f"F:{result.frame_id} {result.latency_ms:.0f}ms"
        cv2.putText(
            annotated, fps_text, (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

        return annotated

    @staticmethod
    def show(frame: np.ndarray) -> bool:
        """
        Exibe o frame e retorna False se o usuário pressionou 'q'.
        Retorna True para continuar.
        """
        if not DISPLAY.show_window:
            return True
        cv2.imshow(DISPLAY.window_name, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
