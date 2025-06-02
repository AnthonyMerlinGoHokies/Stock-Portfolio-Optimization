import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
import logging
import argparse

# ==================== MLFLOW INTEGRATION ====================
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
# ============================================================

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments for MLflow Projects
    parser = argparse.ArgumentParser(description="Portfolio Optimization with MLflow Projects")
    parser.add_argument("--model_type", type=str, default="ridge", help="Model type: ridge, lasso, elastic")
    parser.add_argument("--top_stocks", type=int, default=10, help="Number of top stocks to select")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_path", type=str, default="archive", help="Path to data directory")
    
    args = parser.parse_args()
    
    # Set up paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(CURRENT_DIR, args.data_path)
    OUTPUT_PATH = os.path.join(CURRENT_DIR, 'output')
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # MLflow Projects handles everything - we just need to log
    # Set random seed
    np.random.seed(args.random_seed)
    
    logger.info(f"🚀 Starting Portfolio Optimization via MLflow Projects")
    logger.info(f"   📁 Data Path: {DATA_PATH}")
    logger.info(f"   🤖 Model Type: {args.model_type}")
    logger.info(f"   📈 Top Stocks: {args.top_stocks}")
    logger.info(f"   🎲 Random Seed: {args.random_seed}")
    
    # MLflow Projects automatically provides a run context
    # We can log directly without any setup
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("top_stocks", args.top_stocks)
    mlflow.log_param("random_seed", args.random_seed)
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("mlflow_projects", True)
    
    # 1. DATA LOADING
    logger.info("1. Loading data...")
    try:
        companies = pd.read_csv(os.path.join(DATA_PATH, "sp500_companies.csv"))
        stocks = pd.read_csv(os.path.join(DATA_PATH, "sp500_stocks.csv"))
        
        mlflow.log_param("companies_rows", companies.shape[0])
        mlflow.log_param("stocks_rows", stocks.shape[0])
        
        logger.info(f"✅ Loaded datasets - Companies: {companies.shape}, Stocks: {stocks.shape}")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Error loading data: {e}")
        return
    
    # 2. DATA PREPROCESSING
    logger.info("2. Preprocessing data...")
    stocks['Date'] = pd.to_datetime(stocks['Date'])
    stocks = stocks.dropna(subset=['Adj Close'])
    stocks = stocks.sort_values(['Symbol', 'Date'])
    stocks['Daily_Return'] = stocks.groupby('Symbol')['Adj Close'].pct_change()
    stocks = stocks.dropna(subset=['Daily_Return'])
    
    # Feature engineering
    avg_returns = stocks.groupby('Symbol')['Daily_Return'].mean().reset_index()
    avg_returns.columns = ['Symbol', 'Avg_Return']
    features = pd.merge(companies, avg_returns, on='Symbol')
    features = features.dropna(subset=['Marketcap', 'Revenuegrowth', 'Avg_Return'])
    
    # Prepare features for modeling
    features_encoded = pd.get_dummies(features, columns=['Exchange', 'Sector', 'Industry'], drop_first=True)
    X = features_encoded.drop(columns=['Symbol', 'Shortname', 'Longname', 'City', 'State',
                                      'Country', 'Fulltimeemployees', 'Longbusinesssummary', 'Avg_Return'])
    y = features_encoded['Avg_Return']
    
    # Handle non-numeric columns
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
    
    # Clean and split data
    X_cleaned = X.dropna()
    y_cleaned = y[X.index.isin(X_cleaned.index)]
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=args.random_seed)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log data metrics
    mlflow.log_metric("train_size", X_train_scaled.shape[0])
    mlflow.log_metric("test_size", X_test_scaled.shape[0])
    mlflow.log_metric("feature_count", X_train_scaled.shape[1])
    
    logger.info(f"📊 Data prepared - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    # 3. MODEL TRAINING
    logger.info("3. Training model...")
    
    # Select model based on type
    alphas = np.logspace(-4, 2, 20)
    
    if args.model_type == 'ridge':
        model = RidgeCV(alphas=alphas, cv=5)
    elif args.model_type == 'lasso':
        model = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    elif args.model_type in ['elastic', 'elasticnet']:
        l1_ratios = [.1, .5, .7, .9, .95]
        model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, max_iter=10000)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Log metrics
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    
    # Log model parameters
    if hasattr(model, 'alpha_'):
        mlflow.log_param("alpha", model.alpha_)
    if hasattr(model, 'l1_ratio_'):
        mlflow.log_param("l1_ratio", model.l1_ratio_)
    
    logger.info(f"✅ {args.model_type.upper()} - R²: {test_r2:.6f}, MSE: {test_mse:.6f}")
    
    # 4. PORTFOLIO CONSTRUCTION
    logger.info("4. Creating portfolio...")
    predicted_returns = model.predict(X_test_scaled)
    stock_symbols = features['Symbol'].values[X_test.index]
    
    portfolio_df = pd.DataFrame({
        'Symbol': stock_symbols,
        'PredictedReturn': predicted_returns
    }).sort_values('PredictedReturn', ascending=False).head(args.top_stocks)
    
    portfolio_df['Weight'] = 1/len(portfolio_df)
    
    # Calculate portfolio metrics
    portfolio_return = portfolio_df['PredictedReturn'].mean() * 252  # Annualized
    benchmark_return = features['Avg_Return'].mean() * 252
    outperformance = (portfolio_return - benchmark_return) / benchmark_return * 100
    
    # Log portfolio metrics
    mlflow.log_metric("portfolio_return_annual", portfolio_return)
    mlflow.log_metric("benchmark_return_annual", benchmark_return)
    mlflow.log_metric("outperformance_pct", outperformance)
    
    logger.info(f"📈 Portfolio return: {portfolio_return:.2%} vs Benchmark: {benchmark_return:.2%}")
    logger.info(f"🚀 Outperformance: {outperformance:.2f}%")
    
    # 5. LOG MODEL
    logger.info("5. Logging model...")
    input_example = X_test_scaled[:5]
    signature = mlflow.models.infer_signature(X_test_scaled, y_test_pred)
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )
    
    # 6. SAVE RESULTS
    portfolio_file = os.path.join(OUTPUT_PATH, f"portfolio_{args.model_type}_projects.csv")
    portfolio_df.to_csv(portfolio_file, index=False)
    mlflow.log_artifact(portfolio_file)
    
    logger.info(f"🎉 MLflow Projects run completed!")
    logger.info(f"📁 Results saved to: {OUTPUT_PATH}")
    
    return {
        'test_r2': test_r2,
        'test_mse': test_mse,
        'portfolio_return': portfolio_return,
        'outperformance': outperformance
    }

if __name__ == "__main__":
    main()