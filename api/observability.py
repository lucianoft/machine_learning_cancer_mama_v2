"""Logging estruturado + metricas Prometheus — cobre o requisito de "monitoramento e
logging adequados para tracking de desempenho" sem precisar de infra de nuvem (essa
parte fica so documentada em ARQUITETURA.md, como decidido)."""

import logging
import time
from contextlib import contextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger("api")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def instrument(app: FastAPI) -> None:
    """Expoe /metrics (Prometheus) com latencia/contagem de requests por rota — sem
    precisar reimplementar middleware de timing na mao."""
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@contextmanager
def log_prediction(endpoint: str):
    """Loga label previsto, confianca e latencia de uma predicao — nunca os dados
    clinicos crus da amostra. Uso:

        with log_prediction("predict") as ctx:
            resultado = ...
            ctx["label"] = resultado["label"]
            ctx["confianca"] = max(resultado["probabilidade_benigno"], resultado["probabilidade_maligno"])
    """
    start = time.perf_counter()
    ctx: dict = {}
    try:
        yield ctx
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "endpoint=%s label=%s confianca=%s latencia_ms=%.1f",
            endpoint, ctx.get("label", "?"), ctx.get("confianca", "?"), latency_ms,
        )
