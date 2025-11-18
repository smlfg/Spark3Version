from registry.manager import RegistryManager
import sys

print("🔍 TESTING SYSTEM INTEGRATION...")

try:
    # Test 1: Models
    models = RegistryManager.list_models()
    print(f"✅ Found Models: {models}")
    if "qwen-0.5b" in models:
        meta = RegistryManager.get_model("qwen-0.5b")
        print(f"   -> Loaded Config for: {meta.name} (Type: {meta.model_type})")
    else:
        print("❌ ERROR: qwen-0.5b missing from list!")
        sys.exit(1)

    # Test 2: Datasets
    datasets = RegistryManager.list_datasets()
    print(f"✅ Found Datasets: {datasets}")
    if "stackoverflow" in datasets:
        ds_meta = RegistryManager.get_dataset("stackoverflow")
        print(f"   -> Loaded Metadata for: {ds_meta.name} (Path: {ds_meta.train_path})")
    else:
        print("❌ ERROR: stackoverflow missing from list!")
        sys.exit(1)
        
    print("\n🎉 SUCCESS: Registry is connected to File System!")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    sys.exit(1)

