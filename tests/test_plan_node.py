from cdc_agent.agent import plan_node

def test_plan_node_flu():
    state = {"question": "What are flu symptoms?", "topic": "", "cdc_info": "", "summary": ""}
    result = plan_node(state)
    assert result["topic"] == "flu"

def test_plan_node_covid():
    state = {"question": "Tell me about COVID.", "topic": "", "cdc_info": "", "summary": ""}
    result = plan_node(state)
    assert result["topic"] == "covid"

def test_plan_node_mpox():
    state = {"question": "What is mpox?", "topic": "", "cdc_info": "", "summary": ""}
    result = plan_node(state)
    assert result["topic"] == "monkeypox"
