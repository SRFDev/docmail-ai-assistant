from config.loader import AppConfig
from core.aws_service import AwsService

# Mock config for testing
class MockConfig:
    s3_bucket_name = "srf-docmail-data"

config = MockConfig()
svc = AwsService.get_instance(config)

print("Testing LLM initialization...")
print(f"LLM: {svc.llm.model}")

print("Testing Embedding initialization...")
print(f"Embed: {svc.embed_model.model_name}")

print("✅ Service Layer Ported.")

