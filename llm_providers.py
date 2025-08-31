#!/usr/bin/env python3
"""
LLM Provider Management
Supports OpenAI, OpenRouter and My_API providers with unified interface
"""

import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider:
    """Base LLM provider interface"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = None
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion
        
        Returns:
            Tuple[str, Dict[str, int]]: (response_content, token_usage)
            where token_usage contains 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion using OpenAI"""
        # Extract temperature to avoid duplicate parameter
        temperature = kwargs.pop('temperature', 0)
        
        # Validate model name
        if not self.model_name:
            raise ValueError(f"Model name is empty or None: '{self.model_name}'")
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        
        # Extract token usage information
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens,
            'total_tokens': completion.usage.total_tokens
        }
        
        return completion.choices[0].message.content, token_usage

class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion using OpenRouter"""
        # Extract temperature to avoid duplicate parameter
        temperature = kwargs.pop('temperature', 0)
        
        # Validate model name
        if not self.model_name:
            raise ValueError(f"Model name is empty or None: '{self.model_name}'")
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        
        # Extract token usage information
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens,
            'total_tokens': completion.usage.total_tokens
        }
        
        return completion.choices[0].message.content, token_usage

class MyAPIProvider(LLMProvider):
    """My_API provider implementation with custom base_url and secure API key management"""
    
    # Configuration for different model categories (API keys loaded from environment)
    MODEL_CONFIG = [
        {
            "category": "openai",
            "env_key": "MY_API_OPENAI_KEY",
            "models": ["gpt-4.1", "gpt-4o"]
        },
        {
            "category": "claude",
            "env_key": "MY_API_CLAUDE_KEY",
            "models": ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"]
        },
        {
            "category": "gemini",
            "env_key": "MY_API_GEMINI_KEY",
            "models": ["gemini-2.5-flash"]
        }
    ]
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        # Find the appropriate API key for the model
        api_key = self._get_api_key_for_model(model_name)
        if not api_key:
            raise ValueError(f"No API key found for model: {model_name}. Please check your .env file.")
        
        # Initialize client with custom base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url="http://152.53.170.181:3000/v1"
        )
    
    def _get_api_key_for_model(self, model_name: str) -> str:
        """Get the appropriate API key for the given model from environment variables"""
        for config in self.MODEL_CONFIG:
            if model_name in config["models"]:
                api_key = os.getenv(config["env_key"])
                if not api_key:
                    raise ValueError(f"API key not found in environment variables: {config['env_key']}")
                return api_key
        return None
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate chat completion using My_API"""
        # Extract temperature to avoid duplicate parameter
        temperature = kwargs.pop('temperature', 0)
        
        # Validate model name
        if not self.model_name:
            raise ValueError(f"Model name is empty or None: '{self.model_name}'")
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        
        # Extract token usage information
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens,
            'total_tokens': completion.usage.total_tokens
        }
        
        return completion.choices[0].message.content, token_usage

def create_provider(provider_name: str, model_name: str) -> LLMProvider:
    """Factory function to create LLM provider"""
    if provider_name.lower() == 'openai':
        return OpenAIProvider(model_name)
    elif provider_name.lower() == 'openrouter':
        return OpenRouterProvider(model_name)
    elif provider_name.lower() == 'my_api':
        return MyAPIProvider(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")