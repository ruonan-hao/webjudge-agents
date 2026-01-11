#!/usr/bin/env python3
"""
Test script to verify Nebius endpoint is working.

This script tests:
1. API key and connection
2. Model availability
3. Basic vision-language model call
"""

import os
import sys
import base64
import io
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from openai import OpenAI
except ImportError:
    print("❌ ERROR: OpenAI package not installed")
    print("   Install with: pip install openai")
    sys.exit(1)


def test_nebius_connection():
    """Test basic Nebius API connection."""
    
    # Check for API key
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("❌ ERROR: NEBIUS_API_KEY environment variable not set")
        print("   Please set it in your .env file:")
        print("   NEBIUS_API_KEY=your_api_key_here")
        return False
    
    print("✅ API Key found")
    
    # Get configuration
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    model_name = os.environ.get("NEBIUS_MODEL_NAME", "google/gemma-3-27b-it-fast")
    
    print(f"🔗 Base URL: {base_url}")
    print(f"🤖 Model: {model_name}")
    
    try:
        # Initialize client
        print("\n🔌 Testing API connection...")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Test with a simple text-only call first (matching Gemma-3 template)
        print("📤 Sending test request (text only)...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Say 'Hello, Gemma-3 is working!' if you can read this."
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        message = response.choices[0].message.content
        print(f"✅ API connection successful!")
        print(f"📥 Response: {message}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Connection failed")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        return False


def test_nebius_vision():
    """Test Nebius with a simple image."""
    
    api_key = os.environ.get("NEBIUS_API_KEY")
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    model_name = os.environ.get("NEBIUS_MODEL_NAME", "google/gemma-3-27b-it-fast")
    
    if not api_key:
        print("❌ NEBIUS_API_KEY not set")
        return False
    
    try:
        print("\n🖼️  Testing vision capability...")
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Create a simple test image (red square)
        test_image = Image.new('RGB', (100, 100), color='red')
        buffered = io.BytesIO()
        test_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        print("📤 Sending test request with image...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What color is this image? Just say the color name."
                        }
                    ]
                }
            ],
            max_tokens=20
        )
        
        message = response.choices[0].message.content
        print(f"✅ Vision test successful!")
        print(f"📥 Response: {message}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Vision test failed")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_language_agent():
    """Test VisionLanguageAgent initialization."""
    
    print("\n🤖 Testing VisionLanguageAgent initialization...")
    
    try:
        from vision_language_agent import VisionLanguageAgent
        
        api_key = os.environ.get("NEBIUS_API_KEY")
        base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        model_name = os.environ.get("NEBIUS_MODEL_NAME", "google/gemma-3-27b-it-fast")
        
        if not api_key:
            print("❌ NEBIUS_API_KEY not set")
            return False
        
        agent = VisionLanguageAgent(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )
        
        print("✅ VisionLanguageAgent initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: VisionLanguageAgent initialization failed")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("🧪 Testing Nebius Endpoint")
    print("="*60)
    print()
    
    # Test 1: Basic connection
    print("Test 1: Basic API Connection")
    print("-" * 60)
    connection_ok = test_nebius_connection()
    
    if not connection_ok:
        print("\n❌ Basic connection test failed. Please check your API key and endpoint.")
        sys.exit(1)
    
    # Test 2: Vision capability
    print("\n" + "="*60)
    print("Test 2: Vision Capability")
    print("-" * 60)
    vision_ok = test_nebius_vision()
    
    # Test 3: Agent initialization
    print("\n" + "="*60)
    print("Test 3: VisionLanguageAgent Initialization")
    print("-" * 60)
    agent_ok = test_vision_language_agent()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"✅ API Connection: {'PASS' if connection_ok else 'FAIL'}")
    print(f"✅ Vision Capability: {'PASS' if vision_ok else 'FAIL'}")
    print(f"✅ Agent Initialization: {'PASS' if agent_ok else 'FAIL'}")
    
    if connection_ok and vision_ok and agent_ok:
        print("\n🎉 All tests passed! Nebius endpoint is working correctly.")
        print("\n💡 You can now use it with:")
        print("   WEB_AGENT_TYPE=nebius python scenarios/webjudge/web_agent.py")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

