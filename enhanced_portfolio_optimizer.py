#!/usr/bin/env python3
"""
Enhanced Portfolio Optimizer with MLflow Model Registry Integration
Combines portfolio optimization with comprehensive model tracking and registry management.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPortfolioOptimizer:
    """Enhanced Portfolio Optimizer with MLflow Model Registry integration."""
    
    def __init__(self):
        """Initialize the enhanced portfolio optimizer."""
        # Environment configuration
        self.data_path = os.getenv('DATA_PATH', '/home/optimizer/app/archive')
        self.output_path = os.getenv('OUTPUT_PATH', '/home/optimizer/app/output')
        self.model_type = os.getenv('MODEL_TYPE', 'ridge').lower()
        self.top_stocks = int(os.getenv('TOP_STOCKS', 10))
        self.random_seed = int(os.getenv('RANDOM_SEED', 42))
        
        # MLflow configuration - Connect to server instead of local files
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
        self.experiment_name = "Portfolio_Optimization_Registry"
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_uri)
        logger.info(f"🔬 MLflow tracking URI: {self.mlflow_uri}")
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"🔬 Created new MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"🔬 Using existing MLflow experiment: {self.experiment_name}")
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"❌ Error setting up MLflow experiment: {e}")
            # Fallback to default experiment
            mlflow.set_experiment("Default")
        
        # Initialize MLflow client for model registry operations
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
        
        # Data containers
        self.companies_df = None
        self.stocks_df = None
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        
        # Results storage
        self.results = {}
        
        # Model configurations
        self.model_configs = {
            'ridge': {'model': Ridge, 'params': {'alpha': 1.0, 'random_state': self.random_seed}},
            'lasso': {'model': Lasso, 'params': {'alpha': 0.1, 'random_state': self.random_seed}},
            'elastic': {'model': ElasticNet, 'params': {'alpha': 0.1, 'l1_ratio': 0.5, 'random_state': self.random_seed}},
            'forest': {'model': RandomForestRegressor, 'params': {'n_estimators': 100, 'random_state': self.random_seed}}
        }
        
        logger.info(f"📁 Data path: {self.data_path}")
        logger.info(f"📁 Output path: {self.output_path}")

    def load_data(self):
        """Load and combine stock market datasets."""
        logger.info("1. Loading data...")
        
        try:
            # Check if directory exists
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Directory {self.data_path} does not exist!")
            
            # List available files
            files = os.listdir(self.data_path)
            logger.info(f"📁 Looking in directory: {self.data_path}")
            
            # Load companies data
            companies_file = os.path.join(self.data_path, 'sp500_companies.csv')
            if os.path.exists(companies_file):
                self.companies_df = pd.read_csv(companies_file)
                logger.info(f"✅ Loaded companies data: {self.companies_df.shape}")
            else:
                raise FileNotFoundError(f"Companies file not found: {companies_file}")
            
            # Load stocks data
            stocks_file = os.path.join(self.data_path, 'sp500_stocks.csv')
            if os.path.exists(stocks_file):
                self.stocks_df = pd.read_csv(stocks_file)
                self.stocks_df['Date'] = pd.to_datetime(self.stocks_df['Date'])
                logger.info(f"✅ Loaded stocks data: {self.stocks_df.shape}")
            else:
                raise FileNotFoundError(f"Stocks file not found: {stocks_file}")
            
            # Load index data if available
            index_file = os.path.join(self.data_path, 'sp500_index.csv')
            if os.path.exists(index_file):
                index_df = pd.read_csv(index_file)
                logger.info(f"✅ Loaded index data: {index_df.shape}")
            
            logger.info(f"✅ Loaded datasets - Companies: {self.companies_df.shape}, Stocks: {self.stocks_df.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            logger.info(f"📁 Looking in directory: {self.data_path}")
            if os.path.exists(self.data_path):
                files = os.listdir(self.data_path)
                logger.info(f"📁 Available files: {files}")
            else:
                logger.error(f"Directory {self.data_path} does not exist!")
            return False

    def preprocess_data(self):
        """Preprocess data for model training."""
        logger.info("2. Preprocessing data...")
        
        try:
            # Calculate returns and technical indicators
            stock_features = []
            
            for symbol in self.stocks_df['Symbol'].unique()[:100]:  # Limit for processing
                stock_data = self.stocks_df[self.stocks_df['Symbol'] == symbol].copy()
                stock_data = stock_data.sort_values('Date')
                
                if len(stock_data) < 30:  # Need minimum data
                    continue
                
                # Calculate returns
                stock_data['Returns'] = stock_data['Close'].pct_change()
                
                # Calculate technical indicators
                stock_data['SMA_10'] = stock_data['Close'].rolling(10).mean()
                stock_data['SMA_30'] = stock_data['Close'].rolling(30).mean()
                stock_data['Volatility'] = stock_data['Returns'].rolling(20).std()
                stock_data['Volume_MA'] = stock_data['Volume'].rolling(10).mean()
                
                # Calculate ratios
                stock_data['Price_to_SMA'] = stock_data['Close'] / stock_data['SMA_30']
                stock_data['Volume_Ratio'] = stock_data['Volume'] / stock_data['Volume_MA']
                
                # Get latest features for this stock
                latest = stock_data.iloc[-1]
                recent_data = stock_data.tail(20)
                
                features = {
                    'Symbol': symbol,
                    'Latest_Return': latest['Returns'] if pd.notna(latest['Returns']) else 0,
                    'Avg_Return': recent_data['Returns'].mean(),
                    'Volatility': recent_data['Volatility'].iloc[-1] if pd.notna(recent_data['Volatility'].iloc[-1]) else 0,
                    'Price_to_SMA': latest['Price_to_SMA'] if pd.notna(latest['Price_to_SMA']) else 1,
                    'Volume_Ratio': latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 1,
                    'Price_Range': (latest['High'] - latest['Low']) / latest['Close'],
                    'Close_Price': latest['Close'],
                    'Target_Return': recent_data['Returns'].mean()  # Target variable
                }
                
                stock_features.append(features)
            
            # Create features DataFrame
            self.features_df = pd.DataFrame(stock_features)
            self.features_df = self.features_df.dropna()
            
            # Add company fundamentals if available
            if self.companies_df is not None:
                # Find the sector column (could be 'GICS Sector', 'Sector', etc.)
                sector_col = None
                for col in ['GICS Sector', 'Sector', 'GICS_Sector', 'sector']:
                    if col in self.companies_df.columns:
                        sector_col = col
                        break
                
                if sector_col:
                    company_features = self.companies_df[['Symbol', sector_col]].copy()
                    self.features_df = self.features_df.merge(company_features, on='Symbol', how='left')
                    
                    # Create sector dummies
                    sector_dummies = pd.get_dummies(self.features_df[sector_col], prefix='Sector')
                    self.features_df = pd.concat([self.features_df, sector_dummies], axis=1)
                else:
                    logger.warning("⚠️ No sector column found in companies data")
            
            logger.info(f"📊 Features created: {self.features_df.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in preprocessing: {e}")
            return False

    def prepare_model_data(self):
        """Prepare data for model training."""
        try:
            # Select feature columns (exclude non-numeric and target)
            feature_cols = [col for col in self.features_df.columns 
                          if col not in ['Symbol', 'Target_Return', 'GICS Sector', 'Sector', 'GICS_Sector', 'sector'] 
                          and self.features_df[col].dtype in ['float64', 'int64']]
            
            X = self.features_df[feature_cols].fillna(0)
            y = self.features_df['Target_Return'].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"📊 Data prepared - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
            
        except Exception as e:
            logger.error(f"❌ Error preparing model data: {e}")
            return None, None, None, None, None

    def train_and_register_model(self, X_train, X_test, y_train, y_test, feature_cols):
        """Train model and register it with MLflow."""
        logger.info("3. Training models with registry integration...")
        
        # Get model configuration
        if self.model_type not in self.model_configs:
            logger.warning(f"⚠️ Unknown model type: {self.model_type}, using 'ridge'")
            self.model_type = 'ridge'
        
        config = self.model_configs[self.model_type]
        model_class = config['model']
        model_params = config['params']
        
        logger.info(f"🤖 Training {self.model_type.upper()}...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"portfolio_optimization_{self.model_type}") as run:
            # Train model
            self.model = model_class(**model_params)
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("top_stocks", self.top_stocks)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("n_features", len(feature_cols))
            
            # Log metrics
            mlflow.log_metrics({
                "test_r2": r2,
                "test_mse": mse,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            # Log model with simplified approach
            try:
                # First, just log the model to the run
                logger.info("📦 Logging model to MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    input_example=X_train[:1] if len(X_train) > 0 else None
                )
                logger.info("✅ Model logged to run successfully")
                
                # Then try to register it
                try:
                    model_name = f"Portfolio_Optimizer_{self.model_type}"
                    model_uri = f"runs:/{run.info.run_id}/model"
                    
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name=model_name
                    )
                    version = registered_model.version
                    logger.info(f"✅ Model registered: {model_name} version {version}")
                    
                except Exception as reg_error:
                    logger.warning(f"⚠️ Model registration failed: {reg_error}")
                    version = "logged_only"
                    
            except Exception as log_error:
                logger.error(f"❌ Model logging failed: {log_error}")
                version = "failed"
            
            # Promote to production if performance is good
            if r2 > 0.1:  # Threshold for promotion
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production"
                )
                logger.info(f"🚀 Model {model_name} v{version} promoted to PRODUCTION (R²: {r2:.6f})")
            
            # Store results
            self.results = {
                'model_name': model_name,
                'version': version,
                'r2_score': r2,
                'mse': mse,
                'run_id': run.info.run_id
            }
            
            logger.info(f"✅ {self.model_type.upper()} - R²: {r2:.6f}, MSE: {mse:.6f}")
            
            return r2, mse

    def create_portfolio(self):
        """Create optimized portfolio using the trained model."""
        logger.info("4. Creating portfolio...")
        
        try:
            # Select feature columns (exclude non-numeric and target)
            feature_cols = [col for col in self.features_df.columns 
                          if col not in ['Symbol', 'Target_Return', 'GICS Sector', 'Sector', 'GICS_Sector', 'sector'] 
                          and self.features_df[col].dtype in ['float64', 'int64']]
            
            X_all = self.features_df[feature_cols].fillna(0)
            X_all_scaled = self.scaler.transform(X_all)
            
            # Predict returns
            predicted_returns = self.model.predict(X_all_scaled)
            
            # Create portfolio DataFrame
            portfolio_df = self.features_df[['Symbol']].copy()
            portfolio_df['Predicted_Return'] = predicted_returns
            portfolio_df['Score'] = predicted_returns  # Simple scoring
            
            # Select top stocks
            top_stocks = portfolio_df.nlargest(self.top_stocks, 'Score')
            
            # Equal weight allocation for simplicity
            top_stocks['Weight'] = 1.0 / len(top_stocks)
            
            # Calculate portfolio metrics
            portfolio_return = (top_stocks['Predicted_Return'] * top_stocks['Weight']).sum()
            
            # Benchmark (equal weight of all stocks)
            benchmark_return = predicted_returns.mean()
            
            outperformance = ((portfolio_return - benchmark_return) / benchmark_return * 100) if benchmark_return != 0 else 0
            
            # Log portfolio metrics to MLflow
            with mlflow.start_run(run_id=self.results['run_id']):
                mlflow.log_metrics({
                    "portfolio_return": portfolio_return,
                    "benchmark_return": benchmark_return,
                    "outperformance_pct": outperformance,
                    "portfolio_size": len(top_stocks)
                })
            
            logger.info(f"📈 Portfolio return: {portfolio_return:.2%} vs Benchmark: {benchmark_return:.2%}")
            logger.info(f"🚀 Outperformance: {outperformance:.2f}%")
            
            # Save portfolio
            os.makedirs(self.output_path, exist_ok=True)
            portfolio_file = os.path.join(self.output_path, f'portfolio_{self.model_type}.csv')
            top_stocks.to_csv(portfolio_file, index=False)
            
            return top_stocks, portfolio_return, benchmark_return, outperformance
            
        except Exception as e:
            logger.error(f"❌ Error creating portfolio: {e}")
            return None, None, None, None

    def demonstrate_model_registry(self):
        """Demonstrate model registry features."""
        logger.info("\n5. Model Registry Demonstration:")
        
        try:
            model_name = self.results['model_name']
            
            # Get all versions of the model
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            if versions:
                logger.info(f"\n📊 Model Versions Comparison for {model_name}:")
                
                version_data = []
                for version in versions:
                    version_data.append({
                        'version': version.version,
                        'stage': version.current_stage,
                        'test_r2': 0,  # Would be populated from run metrics in real scenario
                        'test_mse': 0,  # Would be populated from run metrics in real scenario
                        'model_type': self.model_type
                    })
                
                versions_df = pd.DataFrame(version_data)
                logger.info(f"\n{versions_df.to_string(index=False)}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not demonstrate model registry: {e}")

    def run_optimization(self):
        """Run the complete portfolio optimization pipeline."""
        logger.info("🚀 Starting Enhanced Portfolio Optimization with Model Registry")
        logger.info(f"   📁 Data Path: {self.data_path}")
        logger.info(f"   📊 Output Path: {self.output_path}")
        logger.info(f"   🤖 Model Type: {self.model_type}")
        logger.info(f"   📈 Top Stocks: {self.top_stocks}")
        
        try:
            # Load and preprocess data
            if not self.load_data():
                return False
            
            if not self.preprocess_data():
                return False
            
            # Prepare model data
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_model_data()
            if X_train is None:
                return False
            
            # Train and register model
            r2, mse = self.train_and_register_model(X_train, X_test, y_train, y_test, feature_cols)
            
            logger.info(f"🏆 Best model: {self.model_type.upper()} (R²: {r2:.6f})")
            
            # Create portfolio
            portfolio, portfolio_return, benchmark_return, outperformance = self.create_portfolio()
            
            # Demonstrate model registry
            self.demonstrate_model_registry()
            
            logger.info("\n🎉 Enhanced Portfolio Optimization Complete!")
            logger.info(f"🆔 MLflow Run ID: {self.results['run_id']}")
            logger.info(f"🌐 MLflow UI: http://localhost:5001")
            logger.info(f"📁 Results saved to: {self.output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            return False

def main():
    """Main execution function."""
    try:
        # Create and run optimizer
        optimizer = EnhancedPortfolioOptimizer()
        success = optimizer.run_optimization()
        
        if success:
            logger.info("✅ Portfolio optimization completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Portfolio optimization failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()