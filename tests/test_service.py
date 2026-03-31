# tests/test_service.py
from src.service import run_analysis

def test_mean():
    data = type("obj", (), {"values": [1, 2, 3, 4]})
    
    result = run_analysis(data)
    
    assert result["mean"] == 2.5
    assert result["n"] == 4
