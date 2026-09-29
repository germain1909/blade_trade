DEFAULT_CONFIG = {
    "agent_llm_model": "qwen/qwen3-vl-4b",
    "graph_llm_model": "qwen/qwen3-vl-4b",
    "agent_llm_provider": "openai",  # "openai", "anthropic", "qwen", "minimax", or "minimax_cn"
    "graph_llm_provider": "openai",  # "openai", "anthropic", "qwen", "minimax", or "minimax_cn"
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    # Points the "openai" provider at a local OpenAI-compatible server (LM Studio) instead of
    # api.openai.com -- see trading_graph.py's _create_llm. Leave as "" to use real OpenAI.
    "openai_base_url": "http://127.0.0.1:1234/v1",
    "api_key": "lm-studio",  # OpenAI API key -- LM Studio ignores the value but requires non-empty
    "anthropic_api_key": "sk-",  # Anthropic API key (optional, can also use ANTHROPIC_API_KEY env var)
    "qwen_api_key": "sk-",  # Qwen API key (optional, can also use DASHSCOPE_API_KEY env var)
    "minimax_api_key": "",  # MiniMax API key (optional, can also use MINIMAX_API_KEY env var)
    "minimax_cn_api_key": "",  # MiniMax CN API key (optional, can also use MINIMAX_CN_API_KEY or MINIMAX_API_KEY env var)
}
