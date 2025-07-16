#!/usr/bin/env python3
"""
Test script to verify OpenAI API key and determine the correct API base URL.
"""

import os
import sys
from openai import OpenAI

def test_standard_openai():
    """Test with standard OpenAI API endpoint."""
    print("Testing with standard OpenAI API endpoint...")
    try:
        client = OpenAI()
        response = client.models.list()
        print("✅ Success! Your API key works with the standard OpenAI API.")
        print(f"Available models: {[model.id for model in response.data[:5]]}")
        return True
    except Exception as e:
        print(f"❌ Error with standard OpenAI API: {e}")
        return False

def test_with_base_url(base_url):
    """Test with a specific API base URL."""
    print(f"\nTesting with base URL: {base_url}")
    try:
        client = OpenAI(base_url=base_url)
        response = client.models.list()
        print(f"✅ Success! Your API key works with base URL: {base_url}")
        print(f"Available models: {[model.id for model in response.data[:5]]}")
        return True
    except Exception as e:
        print(f"❌ Error with base URL {base_url}: {e}")
        return False

def test_azure_openai(resource_name):
    """Test with Azure OpenAI API."""
    api_version = "2023-05-15"  # Use the latest API version
    base_url = f"https://{resource_name}.openai.azure.com/openai/deployments"
    
    print(f"\nTesting with Azure OpenAI API: {base_url}")
    try:
        # For Azure, we need to use a different client setup
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=api_version,
            azure_endpoint=f"https://{resource_name}.openai.azure.com"
        )
        
        # For Azure, we need to list deployments instead of models
        # This is just a simple test to see if authentication works
        response = client.models.list()
        print("✅ Success! Your API key works with Azure OpenAI API.")
        print(f"Available models: {[model.id for model in response.data[:5]]}")
        return True
    except Exception as e:
        print(f"❌ Error with Azure OpenAI API: {e}")
        return False

def main():
    """Main function to test the API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not found.")
        return
    
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Test with standard OpenAI API
    standard_works = test_standard_openai()
    
    if not standard_works:
        # If standard API doesn't work, try with different base URLs
        print("\nTrying alternative API endpoints...")
        
        # Try with OpenAI Azure
        if len(sys.argv) > 1:
            # If a resource name is provided as an argument
            azure_resource_name = sys.argv[1]
            test_azure_openai(azure_resource_name)
        else:
            print("\nTo test with Azure OpenAI, run: python test_api_key.py YOUR_AZURE_RESOURCE_NAME")
        
        # Try with common alternative base URLs
        test_with_base_url("https://api.openai.com/v1")
        test_with_base_url("https://oai.hconeai.com/v1")
        test_with_base_url("https://api.openai.azure.com/v1")

if __name__ == "__main__":
    main()
