import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Measles Prediction Model Training",
    page_icon="🦠",
    layout="wide"
)

# Define functions for data visualization
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def plot_feature_importance(model, X_train):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X_train.columns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("This model doesn't provide feature importances.")

def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    st.pyplot(fig)

def get_download_link(model, filename="measles_prediction_model.pkl"):
    """Generate a download link for the trained model"""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Trained Model</a>'

# Main app
def main():
    st.title("🦠 Measles Prediction Model Training")
    st.write("""
    This application helps you build and train regression models to predict measles cases.
    Upload your dataset and follow the step-by-step process to develop a machine learning approach for measles prediction.
    """)
    
    # Step 1: Data Upload
    st.header("Step 1: Upload Data")
    st.write("Upload a CSV file containing your measles data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data successfully loaded!")
            
            # Display basic information about the data
            st.subheader("Data Overview")
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Number of features: {df.shape[1]}")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            with st.expander("Data Summary Statistics"):
                st.dataframe(df.describe())
            
            with st.expander("Check for Missing Values"):
                missing_values = df.isnull().sum()
                st.dataframe(pd.DataFrame({
                    'Feature': missing_values.index,
                    'Missing Values': missing_values.values,
                    'Percentage': (missing_values.values / len(df) * 100).round(2)
                }))
            
            # Step 2: Data Preprocessing
            st.header("Step 2: Data Preprocessing")
            
            # Select target variable
            st.subheader("Select Target Variable")
            target_column = st.selectbox("Choose the target variable (measles cases):", df.columns)
            
            # Select features
            st.subheader("Select Features")
            all_features = list(df.columns)
            all_features.remove(target_column)
            selected_features = st.multiselect("Choose features for prediction:", all_features, default=all_features)
            
            if not selected_features:
                st.warning("Please select at least one feature.")
                return
            
            # Create feature and target datasets
            X = df[selected_features]
            y = df[target_column]
            
            # Data splitting
            st.subheader("Data Splitting")
            test_size = st.slider("Test set size (% of data):", 10, 40, 20) / 100
            random_state = st.number_input("Random state (for reproducibility):", 0, 100, 42)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            st.write(f"Training set size: {X_train.shape[0]} samples")
            st.write(f"Test set size: {X_test.shape[0]} samples")
            
            # Step 3: Data Analysis
            st.header("Step 3: Data Analysis")
            
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Distribution", "Correlation", "Target Analysis"])
            
            with analysis_tab1:
                st.subheader("Feature Distributions")
                numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns[:5]  # Limit to 5 columns for visualization
                if len(numeric_cols) > 0:
                    feature_to_plot = st.selectbox("Select feature to view distribution:", numeric_cols)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(X[feature_to_plot], kde=True)
                    plt.title(f'Distribution of {feature_to_plot}')
                    st.pyplot(fig)
                else:
                    st.info("No numeric features available for distribution plot.")
            
            with analysis_tab2:
                st.subheader("Correlation Heatmap")
                plot_correlation_heatmap(df)
            
            with analysis_tab3:
                st.subheader("Target Variable Analysis")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(y, kde=True)
                plt.title(f'Distribution of {target_column}')
                st.pyplot(fig)
                
                st.write("Target Variable Statistics:")
                st.write(pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [y.mean(), y.median(), y.std(), y.min(), y.max()]
                }))
            
            # Step 4: Model Selection and Training
            st.header("Step 4: Model Selection and Training")
            
            # Identify categorical and numerical columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Model selection
            model_options = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Elastic Net": ElasticNet(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            
            selected_model = st.selectbox("Select a regression model:", list(model_options.keys()))
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numerical_cols),
                    ('cat', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_cols)
                ]
            )
            
            # Create model pipeline
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model_options[selected_model])
            ])
            
            # Hyperparameter tuning option
            tune_hyperparams = st.checkbox("Tune hyperparameters with GridSearchCV")
            
            # Training the model
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    if tune_hyperparams:
                        st.info("Setting up hyperparameter grid search...")
                        
                        # Define hyperparameter grids for each model
                        param_grids = {
                            "Linear Regression": {},  # No hyperparameters to tune
                            "Ridge Regression": {
                                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                            },
                            "Lasso Regression": {
                                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                            },
                            "Elastic Net": {
                                'model__alpha': [0.01, 0.1, 1.0],
                                'model__l1_ratio': [0.1, 0.5, 0.9]
                            },
                            "Random Forest": {
                                'model__n_estimators': [50, 100],
                                'model__max_depth': [None, 10, 20]
                            },
                            "Gradient Boosting": {
                                'model__n_estimators': [50, 100],
                                'model__learning_rate': [0.01, 0.1, 0.2]
                            }
                        }
                        
                        param_grid = param_grids[selected_model]
                        
                        if param_grid:  # Skip if empty (like Linear Regression)
                            grid_search = GridSearchCV(
                                model_pipeline,
                                param_grid,
                                cv=5,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1
                            )
                            grid_search.fit(X_train, y_train)
                            best_model = grid_search.best_estimator_
                            st.success(f"Best parameters: {grid_search.best_params_}")
                        else:
                            st.info("No hyperparameters to tune for this model. Training with default parameters.")
                            best_model = model_pipeline.fit(X_train, y_train)
                    else:
                        best_model = model_pipeline.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = best_model.predict(X_test)
                    
                    # Step 5: Model Evaluation
                    st.header("Step 5: Model Evaluation")
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                    col3.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
                    col4.metric("R² Score", f"{r2:.4f}")
                    
                    st.metric("Cross-Validation RMSE", f"{cv_rmse:.4f}")
                    
                    # Visualizations
                    st.subheader("Actual vs Predicted Values")
                    plot_actual_vs_predicted(y_test, y_pred)
                    
                    if hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
                        st.subheader("Feature Importance")
                        # For tree-based models
                        if selected_model in ["Random Forest", "Gradient Boosting"]:
                            feature_names = []
                            if len(numerical_cols) > 0:
                                feature_names.extend(numerical_cols)
                            
                            # Get the one-hot encoded feature names if there are categorical columns
                            if len(categorical_cols) > 0:
                                # Extract the OneHotEncoder
                                preprocessor = best_model.named_steps['preprocessor']
                                ohe = preprocessor.transformers_[1][1].named_steps['onehot']
                                if hasattr(ohe, 'get_feature_names_out'):
                                    cat_features = ohe.get_feature_names_out(categorical_cols)
                                    feature_names.extend(cat_features)
                            
                            # Extract the model
                            model = best_model.named_steps['model']
                            importances = model.feature_importances_
                            
                            # Create DataFrame for feature importance
                            if len(feature_names) == len(importances):
                                feature_importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
                                plt.title('Feature Importance')
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.warning("Feature names and importance scores have different lengths. Cannot display feature importance.")
                    
                    # Step 6: Model Download
                    st.header("Step 6: Save and Download Model")
                    st.markdown(get_download_link(best_model), unsafe_allow_html=True)
                    
                    st.success("Model training completed successfully! You can now download your trained model.")
                    
                    # Step 7: Making Predictions
                    st.header("Step 7: Make Predictions with New Data")
                    st.write("""
                    You can now use your trained model to make predictions. You can either:
                    1. Upload a new CSV file with the same features
                    2. Input values manually for prediction
                    """)
                    
                    prediction_option = st.radio("Select prediction method:", ["Upload File", "Manual Input"])
                    
                    if prediction_option == "Upload File":
                        pred_file = st.file_uploader("Upload CSV with new data for prediction", type="csv")
                        if pred_file is not None:
                            pred_df = pd.read_csv(pred_file)
                            st.write("Preview of uploaded data:")
                            st.dataframe(pred_df.head())
                            
                            missing_cols = set(selected_features) - set(pred_df.columns)
                            if missing_cols:
                                st.error(f"Missing columns in the uploaded file: {missing_cols}")
                            else:
                                try:
                                    pred_df = pred_df[selected_features]
                                    predictions = best_model.predict(pred_df)
                                    
                                    result_df = pd.DataFrame({
                                        'Predicted Measles Cases': predictions
                                    })
                                    
                                    st.write("Prediction results:")
                                    st.dataframe(result_df)
                                    
                                    # Download predictions
                                    csv = result_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="measles_predictions.csv">Download Predictions as CSV</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Error making predictions: {str(e)}")
                    else:  # Manual Input
                        st.subheader("Enter values for prediction")
                        
                        input_data = {}
                        for feature in selected_features:
                            # Determine the appropriate input widget based on data type
                            if feature in numerical_cols:
                                # Get min and max values from training data for numerical features
                                min_val = float(X[feature].min())
                                max_val = float(X[feature].max())
                                mean_val = float(X[feature].mean())
                                
                                input_data[feature] = st.number_input(
                                    f"{feature}:", 
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    step=(max_val - min_val) / 100
                                )
                            else:  # Categorical features
                                unique_values = X[feature].unique().tolist()
                                input_data[feature] = st.selectbox(f"{feature}:", unique_values)
                        
                        if st.button("Make Prediction"):
                            # Create a DataFrame from the input data
                            input_df = pd.DataFrame([input_data])
                            
                            # Make prediction
                            try:
                                prediction = best_model.predict(input_df)[0]
                                st.success(f"Predicted Measles Cases: {prediction:.2f}")
                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing the data: {str(e)}")
    
    # Add helpful information
    with st.expander("About Measles Prediction with Machine Learning"):
        st.write("""
        ### Important Features for Measles Prediction
        
        When developing a machine learning model for measles prediction, consider including these types of features:
        
        1. **Demographic Data**: Population density, age distribution, urbanization level
        2. **Vaccination Coverage**: MMR vaccination rates, immunization campaign data
        3. **Historical Outbreak Data**: Previous measles case counts, seasonal patterns
        4. **Socioeconomic Factors**: Education levels, healthcare access, poverty rates
        5. **Climate Variables**: Temperature, humidity, seasonality
        6. **Public Health Measures**: Quarantine policies, school closures during outbreaks
        7. **Travel and Migration**: International travel volumes, population movement
        8. **Healthcare Infrastructure**: Hospital beds per capita, healthcare worker density
        
        ### Regression Analysis Approaches
        
        Different regression techniques have different strengths:
        
        - **Linear Regression**: Simple, interpretable baseline model
        - **Ridge/Lasso Regression**: Good when you have many correlated features
        - **Random Forest/Gradient Boosting**: Can capture non-linear relationships and interactions
        
        ### Model Evaluation
        
        When evaluating your measles prediction model, consider:
        
        - **RMSE**: Measures the average magnitude of prediction errors
        - **MAE**: Less sensitive to outliers than RMSE
        - **R²**: Indicates how much variance in measles cases is explained by your model
        
        ### Data Preprocessing Tips
        
        - Handle missing values appropriately
        - Normalize numerical features
        - Encode categorical variables
        - Check for and address outliers
        - Consider feature engineering based on domain knowledge
        """)

if __name__ == "__main__":
    main()