"""Exercise the real Streamlit -> API -> local model path (requires a running API)."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui.py"), default_timeout=600).run()
assert not app.exception, app.exception
app.text_area[0].input("Опиши мне алгоритм линейной регрессии").run()
next(button for button in app.button if button.label == "Исследовать").click().run(timeout=600)
assert not app.exception, app.exception
result = app.session_state["result"]
assert result["generation"]["available"]
assert result["generation"]["usage"]["llm_calls"] >= 1
assert "$$" in result["answer"]
assert "MSE" in result["answer"]
assert len(result["articles"]) == len({a["source"] for a in result["articles"]})
report = dict(
    exceptions=len(app.exception),
    model=result["generation"]["model"],
    llm_calls=result["generation"]["usage"]["llm_calls"],
    formula=True,
    mse=True,
    unique_source_cards=True,
    answer=result["answer"],
)
folder = Path("reports/production_upgrade")
folder.mkdir(parents=True, exist_ok=True)
(folder / "ui-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "answer"}))
