# test_api.py
"""
API Testing Script
Tests all endpoints of the Credit Risk Scoring API
Verifies health checks, predictions, metrics, and model information endpoints
"""

import requests
import json

# API configuration
API_URL = "http://localhost:8000"


def test_health():
    """
    Test the health check endpoint
    Verifies that the API is running and all components are loaded
    """
    print("\nTEST 1: Health Endpoint")
    print("-" * 50)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Health check failed"
    print("PASSED")


def test_prediction():
    """
    Test the prediction endpoint
    Sends a sample credit application and verifies the prediction response
    """
    print("\nTEST 2: Prediction Endpoint")
    print("-" * 50)
    
    # Sample credit application data
    sample_data = {
        "status": 0,                    # Account status
        "duration": 12,                 # Loan duration in months
        "credit_history": 2,            # Credit history category
        "purpose": 3,                   # Loan purpose
        "amount": 5000,                 # Loan amount
        "savings": 0,                   # Savings account status
        "employment_duration": 2,       # Employment duration category
        "installment_rate": 2,          # Installment rate
        "personal_status_sex": 2,       # Personal status and sex
        "other_debtors": 0,             # Other debtors/guarantors
        "present_residence": 3,         # Present residence duration
        "property": 1,                  # Property ownership
        "age": 35,                      # Age in years
        "other_installment_plans": 0,   # Other installment plans
        "housing": 1,                   # Housing status
        "number_credits": 1,            # Number of existing credits
        "job": 2,                       # Job category
        "people_liable": 1,             # Number of people liable
        "telephone": 0,                 # Telephone availability
        "foreign_worker": 0,            # Foreign worker status
        "gender": 1                     # Gender (for fairness adjustment)
    }
    
    response = requests.post(f"{API_URL}/predict", json=sample_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Prediction request failed"
    
    # Verify response structure
    result = response.json()
    required_fields = ["prediction", "prediction_label", "probability_default", 
                      "probability_good", "risk_level", "fairness_adjusted"]
    for field in required_fields:
        assert field in result, f"Missing field in response: {field}"
    
    print("PASSED")


def test_metrics():
    """
    Test the metrics endpoint
    Retrieves and verifies model performance and fairness metrics
    """
    print("\nTEST 3: Metrics Endpoint")
    print("-" * 50)
    
    response = requests.get(f"{API_URL}/metrics")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Metrics request failed"
    
    # Verify metrics are present
    result = response.json()
    assert "roc_auc" in result, "Missing ROC AUC metric"
    assert "disparate_impact" in result, "Missing fairness metrics"
    
    print("PASSED")


def test_model_info():
    """
    Test the model information endpoint
    Retrieves and verifies model configuration and metadata
    """
    print("\nTEST 4: Model Info Endpoint")
    print("-" * 50)
    
    response = requests.get(f"{API_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Model info request failed"
    
    # Verify expected fields
    result = response.json()
    assert "model_type" in result, "Missing model type"
    assert "fairness_aware" in result, "Missing fairness awareness flag"
    
    print("PASSED")


def run_all_tests():
    """
    Execute all API tests in sequence
    Provides summary of test results
    """
    print("=" * 50)
    print("CREDIT RISK API TEST SUITE")
    print("=" * 50)
    print(f"API URL: {API_URL}")
    
    tests = [
        ("Health Check", test_health),
        ("Prediction", test_prediction),
        ("Metrics", test_metrics),
        ("Model Info", test_model_info)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\nRESULT: All tests passed successfully")
    else:
        print(f"\nRESULT: {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 50)
        print("CONNECTION ERROR")
        print("=" * 50)
        print(f"Cannot connect to API at {API_URL}")
        print("Please ensure the API is running:")
        print("  uvicorn api.main:app --reload")
        exit(1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        exit(1)