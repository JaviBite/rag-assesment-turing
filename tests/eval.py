"""Mini-arnés de evaluación del chatbot con un set de preguntas "doradas".

No es un test de paso/fallo binario como los de ``tests/test_*.py``: invoca el
grafo completo sobre un golden set (``tests/golden_set.json``) y reporta métricas
agregadas para tener una foto rápida de la calidad end-to-end. Las preguntas
están alineadas con los Starters de la UI (ver ``app/chainlit_app.py``) y cubren
las cuatro rutas del orquestador más memoria, RAG multimodal y fidelidad.

Qué mide por caso:
- **Enrutado**: la ruta elegida por el orquestador vs. la esperada.
- **Recuperación de imágenes**: que el RAG devuelva imágenes cuando aplica.
- **Palabras clave**: que la respuesta contenga el resultado esperado (cálculos,
  nombre recordado, etc.). Tolera separadores de miles (479.001.600 ≈ 479001600).
- **Calidad (LLM-as-judge)**: un veredicto del propio LLM sobre si la respuesta
  cumple un criterio (fidelidad, relevancia...). Es un juez débil porque usa el
  mismo modelo 2B; en un caso real se usaría un modelo superior. Desactivable
  con ``--no-judge``.
- **Latencia** por caso (segundos de pared).

Corre DENTRO del contenedor app (con el stack levantado e ingesta hecha):
    docker compose run --rm app python /srv/app/tests/eval.py
    docker compose run --rm app python /srv/app/tests/eval.py --no-judge

Local (sin Docker, con .venv y servicios en localhost):
    .venv/Scripts/python tests/eval.py

Flags:
    --no-judge   No invoca al LLM-as-judge (más rápido, solo checks deterministas).
    --strict     Devuelve código de salida 1 si algún check determinista falla
                 (útil para CI); por defecto siempre sale 0 (es un informe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Resuelve el paquete ``app`` tanto en el contenedor (/srv) como en local
# (raíz del repo), sin depender de cómo se lance.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in ("/srv", str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import build_graph
from app.llm import get_chat_model

GOLDEN_PATH = Path(__file__).with_name("golden_set.json")

# Candidatos donde buscar una imagen real para el caso de detección. Si no hay
# ninguna, ese caso se omite (igual que test_detector.test_detect_real_image).
_IMAGE_CANDIDATES = [
    *Path("/srv/app/tests").glob("*.jpg"),
    *Path("/srv/app/tests").glob("*.png"),
    settings.images_dir / "sample_detection.jpg",
    *_REPO_ROOT.glob("tests/*.jpg"),
    *_REPO_ROOT.glob("tests/*.png"),
    _REPO_ROOT / "image.jpg",
]


class _Verdict(BaseModel):
    cumple: bool = Field(description="True si la respuesta cumple el criterio")
    motivo: str = Field(description="Justificación breve del veredicto")


# ── helpers de comprobación ──────────────────────────────────────────────────

def _only_digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def _contains_any(answer: str, options: list[str]) -> bool:
    """¿Aparece alguna de las opciones? Tolera separadores de miles en números."""
    low = answer.lower()
    answer_digits = _only_digits(answer)
    for opt in options:
        if opt.lower() in low:
            return True
        opt_digits = _only_digits(opt)
        if opt_digits and opt_digits in answer_digits:
            return True
    return False


def _find_image() -> str | None:
    return next((str(p) for p in _IMAGE_CANDIDATES if p.exists()), None)


def _final_answer(messages: list) -> str:
    msg = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    return str(msg.content) if msg else ""


def _judge(question: str, answer: str, criterio: str) -> _Verdict | None:
    """LLM-as-judge: pide un veredicto estructurado. None si el modelo falla."""
    model = get_chat_model(temperature=0.0).with_structured_output(
        _Verdict, method="json_schema"
    )
    prompt = (
        "Eres un evaluador de calidad. Tu única tarea es decidir si la RESPUESTA "
        "cumple el CRITERIO, dada la PREGUNTA.\n"
        "Reglas:\n"
        "- Evalúa SOLO el criterio; ignora el formato, la longitud y la sección "
        "'Fuentes:' del final.\n"
        "- Si el criterio pide que el asistente reconozca que NO tiene la "
        "información, una respuesta que diga claramente que no la encuentra "
        "CUMPLE el criterio (no la penalices por ello).\n"
        "- cumple=true si la respuesta satisface el criterio; cumple=false si no.\n\n"
        f"PREGUNTA:\n{question}\n\nRESPUESTA:\n{answer}\n\nCRITERIO:\n{criterio}"
    )
    try:
        return model.invoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001 — el juez no debe tumbar la evaluación
        print(f"      (juez no disponible: {exc})")
        return None


# ── ejecución de un caso ─────────────────────────────────────────────────────

def _run_case(graph, case: dict, use_judge: bool) -> dict:
    case_id = case["id"]
    turns = case["turns"]
    thread_id = f"eval-{case_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n── {case_id} — {case.get('descripcion', '')}")

    image_path = None
    if case.get("image") == "auto":
        image_path = _find_image()
        if image_path is None:
            print("  [SKIP] no se encontró imagen para el caso de detección "
                  "(deja un tests/*.jpg para activarlo).")
            return {"id": case_id, "skipped": True}
        print(f"  imagen: {image_path}")

    # Reproduce la conversación; solo se evalúa el último turno.
    result: dict = {}
    elapsed = 0.0
    for i, turn in enumerate(turns):
        is_last = i == len(turns) - 1
        inputs = {
            "messages": [HumanMessage(content=turn)],
            # La imagen solo se adjunta en el turno que la necesita (el último).
            "image_path": image_path if is_last else None,
        }
        t0 = time.perf_counter()
        result = graph.invoke(inputs, config=config)
        if is_last:
            elapsed = time.perf_counter() - t0

    answer = _final_answer(result["messages"])
    route = result.get("route", "?")
    retrieved = result.get("retrieved_image_paths") or []

    checks: dict[str, bool] = {}

    if case.get("expected_route"):
        ok = route == case["expected_route"]
        checks["route"] = ok
        _line(ok, f"ruta == '{case['expected_route']}' (got '{route}')")

    if case.get("expect_images"):
        ok = len(retrieved) > 0
        checks["images"] = ok
        _line(ok, f"recuperó imágenes ({len(retrieved)})")

    if case.get("expect_any"):
        ok = _contains_any(answer, case["expect_any"])
        checks["keywords"] = ok
        _line(ok, f"contiene alguno de {case['expect_any']}")

    judge_ok: bool | None = None
    if use_judge and case.get("judge"):
        verdict = _judge(turns[-1], answer, case["judge"])
        if verdict is not None:
            judge_ok = verdict.cumple
            _line(judge_ok, f"juez: {verdict.motivo[:300]}")

    print(f"  latencia: {elapsed:.1f}s")
    print(f"  respuesta: {answer[:160].replace(chr(10), ' ')}...")

    return {
        "id": case_id,
        "skipped": False,
        "route": route,
        "checks": checks,
        "judge_ok": judge_ok,
        "elapsed": elapsed,
    }


def _line(ok: bool, msg: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")


# ── resumen ──────────────────────────────────────────────────────────────────

def _summary(results: list[dict], use_judge: bool) -> bool:
    ran = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]

    def _rate(key: str) -> tuple[int, int]:
        vals = [r["checks"][key] for r in ran if key in r["checks"]]
        return sum(vals), len(vals)

    print("\n" + "=" * 60)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 60)
    print(f"  Casos ejecutados: {len(ran)}  |  omitidos: {len(skipped)}")

    all_hard_ok = True
    for key, label in (("route", "Enrutado"), ("keywords", "Palabras clave"),
                       ("images", "Recuperación de imágenes")):
        passed, total = _rate(key)
        if total:
            pct = 100 * passed / total
            print(f"  {label:<26} {passed}/{total}  ({pct:.0f}%)")
            all_hard_ok = all_hard_ok and passed == total

    if use_judge:
        judged = [r["judge_ok"] for r in ran if r["judge_ok"] is not None]
        if judged:
            passed = sum(judged)
            pct = 100 * passed / len(judged)
            print(f"  {'Calidad (LLM-as-judge)':<26} {passed}/{len(judged)}  ({pct:.0f}%)")

    latencies = [r["elapsed"] for r in ran]
    if latencies:
        print(f"  Latencia media: {sum(latencies) / len(latencies):.1f}s  "
              f"(máx {max(latencies):.1f}s)")
    if skipped:
        print(f"  Omitidos: {', '.join(r['id'] for r in skipped)}")
    print("=" * 60)
    return all_hard_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación con golden set.")
    parser.add_argument("--no-judge", action="store_true",
                        help="No invocar al LLM-as-judge (solo checks deterministas).")
    parser.add_argument("--strict", action="store_true",
                        help="Salir con código 1 si algún check determinista falla.")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"Cargado golden set: {len(golden)} casos ({GOLDEN_PATH.name})")
    print("Construyendo el grafo...")
    graph = build_graph()

    use_judge = not args.no_judge
    results = [_run_case(graph, case, use_judge) for case in golden]
    all_hard_ok = _summary(results, use_judge)

    if args.strict and not all_hard_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
