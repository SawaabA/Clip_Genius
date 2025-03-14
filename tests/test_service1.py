from backend.services.service1 import perform_action1

def test_perform_action1():
    result = perform_action1({})
    assert result == {"result": "Action 1 performed"}