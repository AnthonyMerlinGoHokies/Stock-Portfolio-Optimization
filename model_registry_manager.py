#!/usr/bin/env python3
"""
Local Model Registry Management Tool for Portfolio Optimization

This script provides utilities to:
- View all registered models
- Compare model versions 
- Manage model stages (None → Staging → Production → Archived)
- Load specific model versions
- Delete old model versions
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from datetime import datetime
import argparse

# ==================== FIXED PATHS FOR LOCAL MAC ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_DIR, 'output')
MLFLOW_URI = f"file://{OUTPUT_PATH}/mlruns"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

class ModelRegistryManager:
    def __init__(self):
        self.client = MlflowClient()
    
    def list_all_registered_models(self):
        """List all registered models in the registry"""
        print("🏛️ REGISTERED MODELS")
        print("=" * 50)
        
        try:
            models = self.client.search_registered_models()
            if not models:
                print("No registered models found.")
                print("💡 Run 'python local_portfolio_optimizer.py' first to create models!")
                return
                
            for model in models:
                print(f"\n📦 Model: {model.name}")
                print(f"   Description: {model.description or 'No description'}")
                print(f"   Created: {datetime.fromtimestamp(model.creation_timestamp/1000)}")
                
                # Get latest versions by stage
                versions = self.client.get_latest_versions(model.name, stages=["None", "Staging", "Production", "Archived"])
                
                if versions:
                    print("   Versions by stage:")
                    for version in versions:
                        print(f"      {version.current_stage}: v{version.version}")
                else:
                    print("   No versions found")
                    
        except Exception as e:
            print(f"Error listing models: {e}")
    
    def compare_model_versions(self, model_name):
        """Compare all versions of a specific model"""
        print(f"\n📊 MODEL VERSIONS COMPARISON: {model_name}")
        print("=" * 60)
        
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No versions found for model: {model_name}")
                return None
            
            comparison_data = []
            for version in versions:
                try:
                    run = self.client.get_run(version.run_id)
                    comparison_data.append({
                        'Version': version.version,
                        'Stage': version.current_stage,
                        'Test R²': round(float(run.data.metrics.get('test_r2', 0)), 6),
                        'Test MSE': round(float(run.data.metrics.get('test_mse', 0)), 6),
                        'Model Type': run.data.params.get('model_type', 'unknown'),
                        'Top Stocks': run.data.params.get('top_stocks', 'N/A'),
                        'Created': datetime.fromtimestamp(version.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M'),
                        'Run ID': version.run_id[:8] + '...'
                    })
                except Exception as e:
                    print(f"Warning: Could not get metrics for version {version.version}: {e}")
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                df = df.sort_values('Test R²', ascending=False)
                print(df.to_string(index=False))
                return df
            else:
                print("No valid comparison data found")
                return None
                
        except Exception as e:
            print(f"Error comparing versions: {e}")
            return None
    
    def transition_model_stage(self, model_name, version, new_stage, archive_existing=False):
        """Transition a model version to a new stage"""
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if new_stage not in valid_stages:
            print(f"❌ Invalid stage: {new_stage}. Must be one of: {valid_stages}")
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=new_stage,
                archive_existing_versions=archive_existing
            )
            print(f"✅ Model {model_name} v{version} transitioned to {new_stage}")
            
            if archive_existing and new_stage == "Production":
                print("📦 Previous production models archived automatically")
            
            return True
            
        except Exception as e:
            print(f"❌ Error transitioning model: {e}")
            return False
    
    def load_model_by_stage(self, model_name, stage="Production"):
        """Load a model by its stage"""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get model version info
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                version = versions[0]
                print(f"✅ Loaded {stage} model: {model_name} v{version.version}")
                return model, version
            else:
                print(f"❌ No {stage} model found for: {model_name}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    
    def promote_best_model(self, model_name):
        """Automatically promote the best performing model to production"""
        print(f"🔍 Finding best model version for: {model_name}")
        
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No versions found for model: {model_name}")
                return False
            
            best_version = None
            best_r2 = -float('inf')
            
            for version in versions:
                try:
                    run = self.client.get_run(version.run_id)
                    test_r2 = float(run.data.metrics.get('test_r2', 0))
                    
                    if test_r2 > best_r2:
                        best_r2 = test_r2
                        best_version = version
                        
                except Exception as e:
                    print(f"Warning: Could not evaluate version {version.version}")
            
            if best_version:
                print(f"🏆 Best model: v{best_version.version} (R²: {best_r2:.6f})")
                
                if best_version.current_stage != "Production":
                    success = self.transition_model_stage(
                        model_name, 
                        best_version.version, 
                        "Production", 
                        archive_existing=True
                    )
                    if success:
                        print(f"🚀 Promoted v{best_version.version} to Production")
                    return success
                else:
                    print(f"✅ v{best_version.version} already in Production")
                    return True
            else:
                print("❌ Could not determine best model")
                return False
                
        except Exception as e:
            print(f"❌ Error promoting model: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Model Registry Management Tool")
    parser.add_argument('action', choices=[
        'list', 'compare', 'promote', 'load', 'transition', 'history'
    ], help='Action to perform')
    
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--version', help='Model version')
    parser.add_argument('--stage', choices=['None', 'Staging', 'Production', 'Archived'], 
                       help='Model stage')
    parser.add_argument('--archive', action='store_true', 
                       help='Archive existing versions when promoting to Production')
    
    args = parser.parse_args()
    
    manager = ModelRegistryManager()
    
    print(f"🔗 MLflow Tracking URI: {MLFLOW_URI}")
    print(f"🌐 MLflow UI: http://localhost:5001\n")
    
    if args.action == 'list':
        manager.list_all_registered_models()
    
    elif args.action == 'compare':
        if not args.model:
            print("❌ --model required for compare action")
            return
        manager.compare_model_versions(args.model)
    
    elif args.action == 'promote':
        if not args.model:
            print("❌ --model required for promote action")
            return
        manager.promote_best_model(args.model)
    
    elif args.action == 'load':
        if not args.model:
            print("❌ --model required for load action")
            return
        
        if args.version:
            model, version_info = manager.load_model_by_version(args.model, args.version)
        else:
            stage = args.stage or "Production"
            model, version_info = manager.load_model_by_stage(args.model, stage)
        
        if model:
            print(f"📊 Model loaded successfully. Type: {type(model)}")
    
    elif args.action == 'transition':
        if not all([args.model, args.version, args.stage]):
            print("❌ --model, --version, and --stage required for transition action")
            return
        manager.transition_model_stage(args.model, args.version, args.stage, args.archive)
    
    elif args.action == 'history':
        if not args.model:
            print("❌ --model required for history action")
            return
        
        history = manager.get_model_performance_history(args.model)
        if history is not None:
            print(f"\n📈 PERFORMANCE HISTORY: {args.model}")
            print("=" * 60)
            print(history.to_string(index=False))

    def load_model_by_version(self, model_name, version):
        """Load a specific model version"""
        try:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get version info
            version_info = self.client.get_model_version(model_name, version)
            print(f"✅ Loaded model: {model_name} v{version} (Stage: {version_info.current_stage})")
            return model, version_info
            
        except Exception as e:
            print(f"❌ Error loading model version: {e}")
            return None, None

    def get_model_performance_history(self, model_name):
        """Get performance history across all versions"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No versions found for model: {model_name}")
                return None
            
            history = []
            for version in versions:
                try:
                    run = self.client.get_run(version.run_id)
                    history.append({
                        'version': int(version.version),
                        'stage': version.current_stage,
                        'test_r2': float(run.data.metrics.get('test_r2', 0)),
                        'test_mse': float(run.data.metrics.get('test_mse', 0)),
                        'created': datetime.fromtimestamp(version.creation_timestamp/1000)
                    })
                except Exception as e:
                    print(f"Warning: Could not get data for version {version.version}")
            
            if history:
                df = pd.DataFrame(history).sort_values('version')
                return df
            else:
                return None
                
        except Exception as e:
            print(f"Error getting performance history: {e}")
            return None


if __name__ == "__main__":
    # If no arguments provided, show interactive menu
    import sys
    if len(sys.argv) == 1:
        print("🏛️ MODEL REGISTRY MANAGER")
        print("=" * 40)
        print("Available commands:")
        print("  python local_model_registry_manager.py list")
        print("  python local_model_registry_manager.py compare --model Portfolio_Optimizer_ridge")
        print("  python local_model_registry_manager.py promote --model Portfolio_Optimizer_ridge")
        print("  python local_model_registry_manager.py load --model Portfolio_Optimizer_ridge --stage Production")
        print("  python local_model_registry_manager.py transition --model Portfolio_Optimizer_ridge --version 1 --stage Production --archive")
        print("  python local_model_registry_manager.py history --model Portfolio_Optimizer_ridge")
        print("\n🌐 MLflow UI: http://localhost:5001")
        
        # Quick interactive demo
        manager = ModelRegistryManager()
        print("\n" + "="*50)
        manager.list_all_registered_models()
    else:
        main()