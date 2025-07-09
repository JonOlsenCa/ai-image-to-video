#!/usr/bin/env python3
"""Test if server is running and accessible"""
import requests
import time

def test_server():
    endpoints = [
        "http://localhost:8000/",
        "http://localhost:8000/gpu-info",
        "http://localhost:8000/test",
        "http://localhost:8000/debug",
        "http://localhost:8000/test-stable-generator"
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"✅ Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
        except requests.exceptions.Timeout:
            print("❌ Timeout - server not responding")
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - server not running")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_server()