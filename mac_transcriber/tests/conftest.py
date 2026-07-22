"""Изоляция окружения для тестов.

В рабочем `.env.local` стоят прод-флаги (`MAC_TRANSCRIBER_REPORT_BACKEND=claude`,
`MAC_TRANSCRIBER_DIARIZE_TRACKS=1`). `service.py` грузит `.env.local` на импорте, а
тесты импортят `service` на уровне модуля — без изоляции эти флаги протекают в
тест-процесс через `os.environ`:
  * `REPORT_BACKEND=claude` уводит генерацию отчётов в очередь (kill-switch) и валит
    тесты AI-отчётов;
  * `DIARIZE_TRACKS=1` включает пер-трековый путь там, где тест ждёт обычный VAD, и
    `diarize_audio` зовётся с `options` (TypeError на моках без этого аргумента).

Снимаем флаги перед каждым тестом. Тест, которому флаг реально нужен, ставит его сам
через `monkeypatch.setenv` — это срабатывает уже после фикстуры.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_report_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAC_TRANSCRIBER_REPORT_BACKEND", raising=False)
    monkeypatch.delenv("MAC_TRANSCRIBER_DIARIZE_TRACKS", raising=False)
