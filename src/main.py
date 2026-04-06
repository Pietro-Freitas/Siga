import logging
import logging.handlers
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from config import DISPLAY, OBSTACLE_CLASSES, OBSTACLE_PRIORITIES, OCR, VOICE
from model_manager import InferenceResult, ModelManager
from speech_engine import SpeechEngine
from vision_engine import FrameRenderer, VisionEngine
from voice_interface import VoiceInterface

def _setup_logging(log_dir: Path = Path("logs")) -> None:
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console: INFO+ (evita flood de DEBUG no terminal)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Arquivo rotativo: DEBUG (útil para entrega do TCC)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "siga.log",
        maxBytes=2 * 1024 * 1024,   # 2MB (reduzido de 5MB para Pi 2)
        backupCount=2,               # 2 backups (reduzido de 3)
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Máquina de estados
# ---------------------------------------------------------------------------

class BuscaEstado(Enum):
    IDLE              = auto()
    AGUARDANDO_OBJETO = auto()
    BUSCANDO          = auto()
    ENCONTRADO        = auto()


@dataclass
class EstadoSIGA:
    """Estado mutável compartilhado entre threads (protegido por lock)."""
    busca:          BuscaEstado = BuscaEstado.IDLE
    objetivo:       str         = ""
    ultimo_ocr:     str         = ""
    ultimo_ocr_ts:  float       = 0.0
    _lock: threading.Lock       = field(default_factory=threading.Lock)

    def transition(self, novo_estado: BuscaEstado, objetivo: str = "") -> None:
        with self._lock:
            self.busca    = novo_estado
            self.objetivo = objetivo
        logger.info("Estado → %s (objetivo='%s')", novo_estado.name, objetivo)


# ---------------------------------------------------------------------------
# Mapeamento de nomes PT→EN para busca por voz
# ---------------------------------------------------------------------------

_NOMES_MAP = {
    "pessoa":    "person",
    "pessoas":   "person",
    "carro":     "car",
    "carros":    "car",
    "bicicleta": "bicycle",
    "onibus":    "bus",
    "ônibus":    "bus",
    "moto":      "motorcycle",
    "motocicleta": "motorcycle",
    "caminhao":  "truck",
    "caminhão":  "truck",
    "cachorro":  "dog",
    "gato":      "cat",
    "banco":     "bench",
}


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

class SigaOrchestrator:
    """
    Cérebro do SIGA — integra todos os subsistemas e implementa
    a lógica de guiagem assistiva.

    Responsabilidades:
    • Inicializar e encerrar todos os módulos de forma ordenada.
    • Processar comandos de voz e atualizar a máquina de estados.
    • Receber resultados de visão e decidir o que comunicar ao usuário.
    • Controlar a janela de visualização (modo debug).
    """

    def __init__(self) -> None:
        self._estado        = EstadoSIGA()
        self._cmd_queue:    queue.Queue[str] = queue.Queue()
        self._frame_queue:  queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._should_exit   = threading.Event()

        # Subsistemas
        self._voz           = VoiceInterface()
        self._mm            = ModelManager()
        self._vision:       Optional[VisionEngine]  = None
        self._stt:          Optional[SpeechEngine]  = None

        # Para controle de spam de detecções comuns
        self._last_detection_ts: dict[str, float] = {}
        self._last_det_cleanup = time.monotonic()

    # ------------------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Inicializa o sistema e entra no loop principal."""
        _setup_logging()
        logger.info("=" * 60)
        logger.info("SIGA — Sistema Inteligente de Guiagem Assistiva")
        logger.info("=" * 60)

        # Registro de sinais para encerramento gracioso (Ctrl+C)
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        if not self._initialize():
            logger.critical("Inicialização falhou. Encerrando.")
            sys.exit(1)

        self._voz.falar_agora("Sistema SIGA iniciado. Diga encontrar para iniciar uma busca.")

        # Loop de processamento de comandos (thread principal)
        try:
            self._command_loop()
        except Exception:
            logger.exception("Exceção não tratada no loop principal.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def _initialize(self) -> bool:
        """Inicializa subsistemas na ordem correta."""
        logger.info("Carregando modelos de IA...")
        if not self._mm.load():
            logger.error("Falha crítica ao carregar modelos.")
            return False

        logger.info("Iniciando motor de voz...")
        self._voz.start()

        logger.info("Iniciando motor de visão...")
        self._vision = VisionEngine(
            model_manager=self._mm,
            on_result=self._on_vision_result,
        )
        self._vision.start()

        logger.info("Iniciando reconhecimento de voz...")
        self._stt = SpeechEngine(command_queue=self._cmd_queue)
        self._stt.start()

        logger.info("Todos os subsistemas inicializados com sucesso.")
        return True

    # ------------------------------------------------------------------
    # Loop de comandos (thread principal = UI thread)
    # ------------------------------------------------------------------

    def _command_loop(self) -> None:
        """
        Loop principal: processa comandos de voz e gerencia a janela OpenCV.
        O OpenCV *precisa* rodar na thread principal no macOS/Windows.
        """
        while not self._should_exit.is_set():
            # Processa todos os comandos pendentes (não-bloqueante)
            self._drain_commands()
            
            # Atualiza a janela se houver um novo frame disponível
            if DISPLAY.show_window:
                try:
                    frame_to_show = self._frame_queue.get_nowait()
                    cv2.imshow(DISPLAY.window_name, frame_to_show)
                except queue.Empty:
                    pass

            # Janela de visualização: apenas verifica o 'q'
            if DISPLAY.show_window:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Tecla 'q' pressionada — encerrando.")
                    break

            # ~30Hz é suficiente para UI polling (reduzido de ~100Hz)
            time.sleep(0.033)

            # Limpeza periódica do cache de detecções
            self._cleanup_detection_cache()

    def _drain_commands(self) -> None:
        """Consome todos os comandos disponíveis na fila."""
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
                self._process_command(cmd)
            except queue.Empty:
                break

    def _process_command(self, text: str) -> None:
        """
        Implementa a máquina de estados de comandos de voz.

        Comandos suportados:
        - "encontrar"   → inicia modo busca
        - "parar"       → cancela busca ativa
        - <qualquer>    → tratado como objeto-alvo quando em AGUARDANDO_OBJETO
        """
        logger.info("Processando comando: '%s'", text)
        estado = self._estado.busca

        if text == "encontrar":
            self._estado.transition(BuscaEstado.AGUARDANDO_OBJETO)
            self._voz.falar("O que deseja localizar?", prioridade=True)
            return

        if text == "parar":
            self._estado.transition(BuscaEstado.IDLE)
            self._voz.falar("Busca cancelada.", prioridade=True)
            return

        if estado == BuscaEstado.AGUARDANDO_OBJETO:
            self._estado.transition(BuscaEstado.BUSCANDO, objetivo=text)
            self._voz.falar(f"Procurando {text}.", prioridade=True)
            return

        logger.debug("Comando '%s' ignorado no estado %s.", text, estado.name)

    # ------------------------------------------------------------------
    # Callback de visão (chamado pela VisionEngine na thread de câmera)
    # ------------------------------------------------------------------

    def _on_vision_result(
        self, frame: np.ndarray, result: InferenceResult
    ) -> None:
        """
        Recebe o resultado de inferência e executa a lógica de negócio:
        - Verifica se o objeto buscado foi encontrado.
        - Anuncia obstáculos importantes.
        - Anuncia textos OCR detectados (anti-spam).
        - Atualiza a janela de visualização (se habilitada).
        """
        estado  = self._estado.busca
        objetivo = self._estado.objetivo

        # --- Lógica de busca ---
        if estado == BuscaEstado.BUSCANDO and objetivo:
            for det in result.detections:
                if self._nomes_coincidem(det.name, objetivo):
                    self._estado.transition(BuscaEstado.ENCONTRADO)
                    self._voz.falar(
                        f"{objetivo.capitalize()} encontrado!",
                        prioridade=True,
                    )
                    break

        # --- OCR anti-spam ---
        for texto in result.ocr_texts:
            if not texto:
                continue
            agora = time.monotonic()
            ultimo = self._estado.ultimo_ocr_ts
            if (texto != self._estado.ultimo_ocr
                    or (agora - ultimo) > OCR.spam_interval):
                self._estado.ultimo_ocr    = texto
                self._estado.ultimo_ocr_ts = agora
                logger.info("OCR detectado: '%s'", texto)
                self._voz.falar(texto)

        # --- Detecções de obstáculos (anti-spam genérico) ---
        self._announce_obstacles(result)

        # --- Renderização (Envia para a thread principal) ---
        if DISPLAY.show_window:
            annotated = FrameRenderer.render(frame, result)
            try:
                # Limpa o frame antigo se houver, para manter apenas o mais novo
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(annotated)
            except (queue.Full, queue.Empty):
                pass

    def _announce_obstacles(self, result: InferenceResult) -> None:
        """
        Anuncia obstáculos relevantes com controle de spam por classe.
        As detecções são ordenadas pela prioridade configurada.
        Itens mega perigosos (prioridade <= 4) têm spam reduzido e usam fala prioritária.
        """
        now = time.monotonic()

        # Filtra apenas o que está no config e ordena
        valid_dets = [d for d in result.detections if d.name in OBSTACLE_CLASSES]
        valid_dets.sort(key=lambda d: OBSTACLE_PRIORITIES.get(d.name, 99))

        for det in valid_dets:
            prioridade_nivel = OBSTACLE_PRIORITIES.get(det.name, 99)
            eh_mega_perigoso = prioridade_nivel <= 4
            
            # Reduz bastante o spam_interval para coisas perigosas (ex. 1.5s em vez do padrao)
            spam_espera = 1.5 if eh_mega_perigoso else VOICE.spam_interval
            
            last = self._last_detection_ts.get(det.name, 0.0)
            if now - last > spam_espera:
                self._last_detection_ts[det.name] = now
                self._voz.falar(f"{det.name} detectado.", prioridade=eh_mega_perigoso)

    def _cleanup_detection_cache(self) -> None:
        """Remove entradas expiradas do cache de detecções (evita memory leak)."""
        now = time.monotonic()
        if now - self._last_det_cleanup < VOICE.spam_cleanup_interval:
            return
        self._last_det_cleanup = now
        expired = [
            k for k, v in self._last_detection_ts.items()
            if (now - v) > VOICE.spam_interval * 3
        ]
        for k in expired:
            del self._last_detection_ts[k]

    @staticmethod
    def _nomes_coincidem(detectado: str, objetivo: str) -> bool:
        """
        Correspondência flexível entre nome detectado e objetivo do usuário.
        Ex: objetivo="pessoa" → detectado="person" → True.
        Suporta singular/plural e variações com acento.
        """
        alvo = _NOMES_MAP.get(objetivo.lower(), objetivo.lower())
        return detectado.lower() == alvo

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Sinal %d recebido — encerrando...", signum)
        self._should_exit.set()

    def _shutdown(self) -> None:
        """Encerra todos os subsistemas de forma ordenada."""
        logger.info("Iniciando encerramento ordenado...")

        if self._vision:
            self._vision.stop()
        if self._stt:
            self._stt.stop()

        self._voz.falar_agora("Sistema SIGA encerrado.")
        self._voz.stop()

        cv2.destroyAllWindows()
        logger.info("SIGA encerrado com sucesso.")

if __name__ == "__main__":
    SigaOrchestrator().run()