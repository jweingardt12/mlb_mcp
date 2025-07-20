#!/usr/bin/env python3
"""Simple health check for the MLB MCP server"""
import requests
import sys
import os

def test_health():
    port = os.environ.get('PORT', '8000')
    base_url = f"http://localhost:{port}"
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Root endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"\nHealth endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test tools endpoint
        response = requests.get(f"{base_url}/tools", timeout=5)
        print(f"\nTools endpoint status: {response.status_code}")
        print(f"Number of tools: {len(response.json().get('tools', []))}")
        
        return True
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_health()
    sys.exit(0 if success else 1)