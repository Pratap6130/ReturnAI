#!/usr/bin/env python3
"""
Deployment verification script
Tests all API endpoints to ensure the deployment is working correctly
"""
import sys
import requests
import json
from typing import Dict, Any


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def test_health_check(base_url: str) -> bool:
    """Test the health check endpoint"""
    print_info("Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            model_loaded = data.get("model_loaded", False)
            if model_loaded:
                print_success(f"Health check passed - Model is loaded")
            else:
                print_warning(f"Health check passed but model is NOT loaded")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False


def test_register(base_url: str) -> bool:
    """Test user registration"""
    print_info("Testing user registration...")
    try:
        test_user = {
            "name": "Test User",
            "userid": f"testuser_{hash(base_url) % 10000}",
            "password": "testpass123"
        }
        response = requests.post(
            f"{base_url}/register",
            json=test_user,
            timeout=10
        )
        if response.status_code == 200:
            print_success("Registration endpoint working")
            return True
        elif response.status_code == 400:
            print_warning("Registration endpoint working (user may already exist)")
            return True
        else:
            print_error(f"Registration failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Registration test failed: {str(e)}")
        return False


def test_prediction(base_url: str) -> bool:
    """Test prediction endpoint"""
    print_info("Testing prediction endpoint...")
    try:
        test_data = {
            "product_category": "Electronics",
            "product_price": 299.99,
            "order_quantity": 1,
            "user_age": 28,
            "user_gender": "Male",
            "payment_method": "Credit Card",
            "shipping_method": "Express",
            "discount_applied": 30.0
        }
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if all(key in data for key in ["prediction_label", "probability", "risk_level"]):
                print_success(f"Prediction endpoint working - Risk: {data['risk_level']}, Probability: {data['probability']:.2%}")
                return True
            else:
                print_error("Prediction response missing required fields")
                return False
        else:
            print_error(f"Prediction failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Prediction test failed: {str(e)}")
        return False


def test_recent_predictions(base_url: str) -> bool:
    """Test recent predictions endpoint"""
    print_info("Testing recent predictions endpoint...")
    try:
        response = requests.get(
            f"{base_url}/predictions/recent?limit=5",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print_success(f"Recent predictions endpoint working ({len(data)} predictions found)")
                return True
            else:
                print_error("Recent predictions returned non-list response")
                return False
        else:
            print_error(f"Recent predictions failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Recent predictions test failed: {str(e)}")
        return False


def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("🚀 Return Risk Prediction - Deployment Verification")
    print("="*60 + "\n")
    
    # Get backend URL
    if len(sys.argv) > 1:
        backend_url = sys.argv[1].rstrip('/')
    else:
        print("Usage: python verify_deployment.py <backend_url>")
        print("Example: python verify_deployment.py https://returnai-backend.onrender.com")
        backend_url = input("\nEnter your backend URL: ").strip().rstrip('/')
    
    if not backend_url:
        print_error("No backend URL provided")
        sys.exit(1)
    
    print_info(f"Testing backend at: {backend_url}\n")
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_check(backend_url)),
        ("User Registration", lambda: test_register(backend_url)),
        ("Prediction", lambda: test_prediction(backend_url)),
        ("Recent Predictions", lambda: test_recent_predictions(backend_url)),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
        print()
    
    # Summary
    print("="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.RESET} - {test_name}")
    
    print(f"\n{Colors.BLUE}Results: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print_success("All tests passed! ✨ Your deployment is working correctly.")
        sys.exit(0)
    else:
        print_error(f"{total - passed} test(s) failed. Please check the logs above.")
        print_info("See DEPLOYMENT.md for troubleshooting tips.")
        sys.exit(1)


if __name__ == "__main__":
    main()
