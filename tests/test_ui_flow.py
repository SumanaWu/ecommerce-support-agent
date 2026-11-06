from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import service as service_module
from tests.test_actions import business
from tests.test_flow import runner


def test_chat_renders_confirmation_in_turn_and_disables_input(business, tmp_path, monkeypatch):
    service = service_module.SupportService(tmp_path / 'memory.sqlite')
    monkeypatch.setattr(service_module, 'SupportService', lambda: service)
    monkeypatch.setattr(service_module, 'run_orchestrator', runner)
    monkeypatch.setattr(service_module, 'get_llm', lambda: None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.chat_input[0].set_value('申请退款').run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert len(app.chat_message) == 2
    next(button for button in app.button if button.label == '确认办理').click().run()
    assert not app.exception
    assert app.chat_input[0].disabled
    next(button for button in app.button if button.label == '取消本项').click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
    assert len(app.chat_message) == 2
    service.close()
