from backend.services.service2 import perform_action2

def test_perform_action2():
    result = perform_action2({})
    assert result == {"result": "Action 2 performed"}