#!/usr/bin/env python3
"""
Model Registry Demo - Simple demonstration of MLflow Model Registry features

This script shows you:
1. How models get registered automatically
2. How to compare model versions
3. How to load models by stage or version
4. How to transition models between stages
"""

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime

# Setup MLflow (same as your current setup)
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')
MLFLOW_URI = f"file://{OUTPUT_PATH}/mlruns"
mlflow.set_tracking_uri(MLFLOW_URI)

print("🎯 MODEL REGISTRY DEMONSTRATION")
print("=" * 50)
print(f"🔗 MLflow URI: {MLFLOW_URI}")
print(f"🌐 MLflow UI: http://localhost:5001")
print()

# Initialize MLflow client
client = MlflowClient()

def demo_list_registered_models():
    """Demo: List all registered models"""
    print("1️⃣ LISTING ALL REGISTERED MODELS")
    print("-" * 40)
    
    try:
        models = client.search_registered_models()
        
        if not models:
            print("❌ No registered models found yet.")
            print("💡 Run your portfolio optimizer first to create registered models!")
            return False
        
        for model in models:
            print(f"📦 Model: {model.name}")
            print(f"   Created: {datetime.fromtimestamp(model.creation_timestamp/1000)}")
            
            # Get all versions
            versions = client.search_model_versions(f"name='{model.name}'")
            if versions:
                print(f"   Versions: {len(versions)}")
                
                # Show latest version in each stage
                stages = {}
                for version in versions:
                    stage = version.current_stage
                    if stage not in stages or int(version.version) > int(stages[stage]):
                        stages[stage] = version.version
                
                for stage, version in stages.items():
                    print(f"      {stage}: v{version}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_compare_versions(model_name):
    """Demo: Compare versions of a specific model"""
    print(f"2️⃣ COMPARING VERSIONS: {model_name}")
    print("-" * 40)
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"❌ No versions found for: {model_name}")
            return
        
        print(f"Found {len(versions)} versions:")
        print()
        
        comparison_data = []
        for version in versions:
            try:
                run = client.get_run(version.run_id)
                
                # Extract key metrics
                test_r2 = run.data.metrics.get('test_r2', 0)
                test_mse = run.data.metrics.get('test_mse', 0)
                model_type = run.data.params.get('model_type', 'unknown')
                
                comparison_data.append({
                    'Version': version.version,
                    'Stage': version.current_stage,
                    'Model Type': model_type,
                    'Test R²': f"{test_r2:.6f}",
                    'Test MSE': f"{test_mse:.6f}",
                    'Created': datetime.fromtimestamp(version.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M')
                })
                
            except Exception as e:
                print(f"⚠️ Could not get data for version {version.version}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
        else:
            print("❌ No valid version data found")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_load_model(model_name, stage="Production"):
    """Demo: Load a model by stage"""
    print(f"3️⃣ LOADING {stage.upper()} MODEL: {model_name}")
    print("-" * 40)
    
    try:
        # Check if there's a model in the requested stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not versions:
            print(f"❌ No {stage} model found for: {model_name}")
            
            # Show available stages
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if all_versions:
                available_stages = set(v.current_stage for v in all_versions)
                print(f"💡 Available stages: {', '.join(available_stages)}")
            
            return None
        
        # Load the model
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        version_info = versions[0]
        print(f"✅ Successfully loaded model:")
        print(f"   Model: {model_name}")
        print(f"   Version: {version_info.version}")
        print(f"   Stage: {version_info.current_stage}")
        print(f"   Model Type: {type(model).__name__}")
        
        # Get run information
        try:
            run = client.get_run(version_info.run_id)
            test_r2 = run.data.metrics.get('test_r2', 'N/A')
            print(f"   Test R²: {test_r2}")
        except:
            pass
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def demo_transition_stage(model_name, version, new_stage):
    """Demo: Transition a model to a new stage"""
    print(f"4️⃣ TRANSITIONING MODEL STAGE")
    print("-" * 40)
    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"New Stage: {new_stage}")
    print()
    
    try:
        # Get current stage
        version_info = client.get_model_version(model_name, version)
        current_stage = version_info.current_stage
        
        print(f"Current stage: {current_stage}")
        
        if current_stage == new_stage:
            print(f"✅ Model is already in {new_stage} stage")
            return True
        
        # Transition the model
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=new_stage,
            archive_existing_versions=(new_stage == "Production")
        )
        
        print(f"✅ Successfully transitioned to {new_stage}")
        
        if new_stage == "Production":
            print("📦 Previous production models archived automatically")
        
        return True
        
    except Exception as e:
        print(f"❌ Error transitioning model: {e}")
        return False

def main():
    """Run the complete demo"""
    
    # 1. List all registered models
    has_models = demo_list_registered_models()
    print()
    
    if not has_models:
        print("🚀 TO CREATE REGISTERED MODELS:")
        print("Run your portfolio optimizer with:")
        print("   python enhanced_portfolio_optimizer.py")
        print("   # OR")
        print("   docker-compose up portfolio-optimizer")
        return
    
    # Get first available model for demo
    try:
        models = client.search_registered_models()
        if models:
            demo_model_name = models[0].name
            print(f"🎯 Using model '{demo_model_name}' for demonstration\n")
            
            # 2. Compare versions
            demo_compare_versions(demo_model_name)
            print()
            
            # 3. Load production model (if exists)
            production_model = demo_load_model(demo_model_name, "Production")
            print()
            
            # If no production model, try staging
            if production_model is None:
                staging_model = demo_load_model(demo_model_name, "Staging")
                print()
            
            # 4. Demo stage transition
            versions = client.search_model_versions(f"name='{demo_model_name}'")
            if versions:
                # Find a version not in production
                non_prod_version = None
                for v in versions:
                    if v.current_stage != "Production":
                        non_prod_version = v
                        break
                
                if non_prod_version:
                    demo_transition_stage(demo_model_name, non_prod_version.version, "Staging")
                    print()
            
            print("🎉 MODEL REGISTRY DEMO COMPLETE!")
            print("💡 Key Benefits:")
            print("   ✅ Centralized model storage")
            print("   ✅ Version tracking and comparison")
            print("   ✅ Stage-based model lifecycle")
            print("   ✅ Easy model loading by stage/version")
            print("   ✅ Safe production model updates")
            
        else:
            print("❌ No models found to demonstrate")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    main()