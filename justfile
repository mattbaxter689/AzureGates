set dotenv-load := true

# Submit the DSL pipeline to Azure ML
run:
    uv run orchestrator.py
