#!/usr/bin/env python3
import time
import subprocess
import requests
import sys

def test_startup_time():
    print("Testing server startup time...")
    start_time = time.time()
    
    # Start the server
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to be ready
    max_wait = 10  # seconds
    waited = 0
    ready = False
    
    while waited < max_wait:
        try:
            resp = requests.get("http://localhost:8001/health")
            if resp.status_code == 200:
                ready = True
                break
        except:
            pass
        
        time.sleep(0.1)
        waited += 0.1
    
    startup_duration = time.time() - start_time
    
    if ready:
        print(f"✓ Server started in {startup_duration:.2f} seconds")
        
        # Test tools list endpoint
        tool_start = time.time()
        try:
            resp = requests.post("http://localhost:8001/tools/list", json={})
            tool_duration = time.time() - tool_start
            print(f"✓ Tools list responded in {tool_duration:.2f} seconds")
            print(f"  Found {len(resp.json().get('tools', []))} tools")
        except Exception as e:
            print(f"✗ Tools list failed: {e}")
    else:
        print(f"✗ Server failed to start within {max_wait} seconds")
    
    # Cleanup
    proc.terminate()
    proc.wait()
    
    return startup_duration < 3.0  # Target: under 3 seconds

if __name__ == "__main__":
    success = test_startup_time()
    sys.exit(0 if success else 1)