# Import required libraries
import streamlit as st  # Web app framework
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # Model evaluation and tuning
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Linear regression models
from sklearn.tree import DecisionTreeRegressor  # Decision tree algorithm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Ensemble methods
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Model evaluation metrics
import plotly.graph_objects as go  # Advanced plotting
import plotly.express as px  # Quick plotting
from statsmodels.tsa.seasonal import seasonal_decompose  # Time series decomposition
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Exponential smoothing forecasting
from statsmodels.tsa.arima.model import ARIMA  # ARIMA forecasting model
from scipy import stats  # Statistical functions for normality tests
import warnings



# Replacement for ml_rev.small_footnote
def small_footnote(text):
    st.markdown(f"<sub style='color:gray'>{text}</sub>", unsafe_allow_html=True)

# Example usage
st.write("Here is a chart")
small_footnote("This is a small footnote below the chart")


warnings.filterwarnings('ignore')  # Suppress warning messages

# Configure Streamlit page settings
st.set_page_config(page_title="Sales Prediction Analytics", layout="wide")  # Set page title and wide layout

# Display main title
st.title(" Walmart Weekly- Sales Prediction & Time Series Analysis")
st.markdown("---")  # Add horizontal line

# Create sidebar for user inputs
st.sidebar.header("⚙️ Configuration")

# Radio button to select type of analysis
analysis_type = st.sidebar.radio("Select Analysis Type",
                                 ["Regression Models", "Time Series Analysis", "Both"])

# File uploader widget - allows CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])


# Function to generate sample sales data for demonstration
def generate_sample_data():
    np.random.seed(42)  # Set random seed for reproducibility

    # Create date range for 100 days
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')

    # Create DataFrame with synthetic sales data
    data = pd.DataFrame({
        'date': dates,  # Date column
        # Sales with trend and random variation
        'sales': 1000 + np.cumsum(np.random.randn(100) * 50) + np.arange(100) * 2,
        # Marketing spend with random variation
        'marketing_spend': 200 + np.random.randn(100) * 30,
        # Product price with random variation
        'price': 50 + np.random.randn(100) * 5,
        # Competitor price with random variation
        'competitor_price': 55 + np.random.randn(100) * 5,
        # Day of week (0=Monday, 6=Sunday)
        'day_of_week': [d.dayofweek for d in dates],
        # Month number (1-12)
        'month': [d.month for d in dates]
    })
    return data


# Load data based on user choice
if uploaded_file is not None:
    # If user uploaded a file, read it as CSV
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")  # Show success message
else:
    # If no file uploaded, offer to generate sample data
    if st.sidebar.button("🎲 Generate Sample Data"):
        df = generate_sample_data()  # Call function to create sample data
        st.sidebar.info("Sample data generated!")  # Show info message
    else:
        df = None  # No data available yet

# Main content - only executes if data is available
if df is not None:

    # Create expandable section to view data
    with st.expander("📋 View Data"):
        st.dataframe(df.head(10))  # Display first 10 rows
        # Show dataset dimensions
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")



    # Feature selection section in sidebar
    st.sidebar.subheader("Feature Selection")

    # Get all numeric columns from dataset
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Select target variable (what we want to predict)
    target = st.sidebar.selectbox("Select Target Variable", numeric_cols,
                                  index=0 if 'sales' in numeric_cols else 0)

    # Get feature columns (all numeric except target)
    feature_cols = [col for col in numeric_cols if col != target]

    # Multi-select widget for choosing features
    features = st.sidebar.multiselect("Select Features", feature_cols,
                                      default=feature_cols[:3] if len(feature_cols) >= 3 else feature_cols)

    # Check if at least one feature is selected
    if len(features) > 0:

        # ============ EXPLORATORY DATA ANALYSIS SECTION ============
        st.header("🔍 Exploratory Data Analysis (EDA)")

        with st.expander("📊 Statistical Summary"):
            # Display descriptive statistics for all numeric columns
            st.dataframe(df[features + [target]].describe())

        # ===== CORRELATION ANALYSIS & MULTICOLLINEARITY =====
        st.subheader("🔗 Correlation Analysis & Multicollinearity Check")

        # Calculate correlation matrix
        correlation_matrix = df[features + [target]].corr()

        # Create heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(title='Correlation Heatmap', height=500)
        st.plotly_chart(fig, use_container_width=True)
        small_footnote("The heatmap shows mostly weak correlations among the variables. Store has a moderate negative relationship(-0.34) with Weekly_Sales, while Temperature and Holiday_Flag show minimal influence on sales. No strong linear associations are present.Therefore,t here is no sign of abnormal  Multicollinearity")

        # Check for multicollinearity (VIF - Variance Inflation Factor simulation)
        st.subheader("⚠️ Multicollinearity Detection")

        # Calculate correlation between features (excluding target)
        feature_corr = df[features].corr()

        # Find high correlations (> 0.8)
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i + 1, len(feature_corr.columns)):
                if abs(feature_corr.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'Feature 1': feature_corr.columns[i],
                        'Feature 2': feature_corr.columns[j],
                        'Correlation': round(feature_corr.iloc[i, j], 3)
                    })

        if high_corr_pairs:
            st.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (|correlation| > 0.8)")
            st.dataframe(pd.DataFrame(high_corr_pairs))
            st.info(
                "💡 High correlation between features may indicate multicollinearity. Consider removing one feature from each pair.")
        else:
            st.success("✅ No severe multicollinearity detected (all feature correlations < 0.8)")

        # ===== SCATTER PLOTS - RELATIONSHIP ANALYSIS =====
        st.subheader("📈 Feature Relationships with Target")

        # Create scatter plots for each feature vs target
        num_features = len(features)
        cols_per_row = 2
        num_rows = (num_features + cols_per_row - 1) // cols_per_row

        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                feature_idx = row * cols_per_row + col_idx
                if feature_idx < num_features:
                    with cols[col_idx]:
                        feature = features[feature_idx]

                        # Create scatter plot with trend line
                        fig = px.scatter(df, x=feature, y=target,
                                         trendline="ols",  # Ordinary Least Squares trend line
                                         title=f'{target} vs {feature}',
                                         labels={feature: feature, target: target})

                        # Calculate correlation coefficient
                        corr = df[feature].corr(df[target])
                        fig.add_annotation(
                            text=f'Correlation: {corr:.3f}',
                            xref="paper", yref="paper",
                            x=0.05, y=0.95,
                            showarrow=False,
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1
                        )

                        st.plotly_chart(fig, use_container_width=True)

        # ===== BOX PLOTS - OUTLIER DETECTION =====
        st.subheader("📦 Box Plots - Outlier Detection")

        # Create box plots for all numeric variables
        numeric_cols_with_target = features + [target]

        # Display box plots in grid
        for row in range((len(numeric_cols_with_target) + cols_per_row - 1) // cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                var_idx = row * cols_per_row + col_idx
                if var_idx < len(numeric_cols_with_target):
                    with cols[col_idx]:
                        variable = numeric_cols_with_target[var_idx]

                        # Create box plot
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=df[variable],
                            name=variable,
                            boxmean='sd',  # Show mean and standard deviation
                            marker_color='lightblue'
                        ))
                        fig.update_layout(
                            title=f'Box Plot: {variable}',
                            yaxis_title=variable,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate outlier statistics using IQR method
                        Q1 = df[variable].quantile(0.25)
                        Q3 = df[variable].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[variable] < lower_bound) | (df[variable] > upper_bound)]

                        if len(outliers) > 0:
                            st.caption(f"⚠️ {len(outliers)} outliers detected ({len(outliers) / len(df) * 100:.1f}%)")
                        else:
                            st.caption("✅ No outliers detected")
        small_footnote(
            "The boxplot for Weekly_Sales shows a wide distribution with several high-value outliers, indicating significant variability in weekly revenue across stores. Most sales observations fall within a moderate range, while the extreme upper values reflect occasional peak-sales periods. In contrast, the Holiday_Flag boxplot shows minimal variation, with values concentrated at 0 and a small number at 1, confirming that holiday weeks occur infrequently in the dataset.")

        # ===== DISTRIBUTION PLOTS =====
        st.subheader("📊 Distribution Analysis")

        # Create histograms for all numeric variables
        for row in range((len(numeric_cols_with_target) + cols_per_row - 1) // cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                var_idx = row * cols_per_row + col_idx
                if var_idx < len(numeric_cols_with_target):
                    with cols[col_idx]:
                        variable = numeric_cols_with_target[var_idx]

                        # Create histogram with KDE overlay
                        fig = px.histogram(df, x=variable,
                                           marginal="box",  # Add box plot on top
                                           title=f'Distribution: {variable}',
                                           nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
        small_footnote("From the distribution plots above, Weekly_Sales shows a more prominent and asymmetrical spread of data points, indicating a positive skew. This suggests that the majority of revenue values remained below 3 million during the period from 2010 to 2013")
        # ============ REGRESSION MODELS SECTION ============
        if analysis_type in ["Regression Models", "Both"]:
            st.header("🤖 Regression Models")

            # Prepare feature matrix (X) and target vector (y)
            X = df[features]  # Independent variables
            y = df[target]  # Dependent variable
            # ======================  PREPROCESSING PIPELINE  ======================

            st.header("🧹 Data Preprocessing")

            # -------------------- 1. HANDLE MISSING VALUES --------------------
            st.subheader("🔧 Handling Missing Values")
            missing_counts = df.isnull().sum()

            st.write("Missing values in dataset:")
            st.dataframe(missing_counts)

            if missing_counts.sum() > 0:
                df.fillna(df.median(numeric_only=True), inplace=True)
                st.success("Missing values handled using Median Imputation")

            # -------------------- 2. OUTLIER TREATMENT --------------------
            st.subheader("🚨 Outlier Detection & Treatment")

            outlier_method = st.selectbox(
                "Select Outlier Treatment Method",
                ["None", "IQR (Interquartile Range)", "Z-Score"]
            )

            if outlier_method == "IQR (Interquartile Range)":
                for col in features:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df[col] = np.where(df[col] < lower, lower,
                                       np.where(df[col] > upper, upper, df[col]))
                st.success("Outliers capped using IQR method")

            elif outlier_method == "Z-Score":
                for col in features:
                    z = np.abs(stats.zscore(df[col]))
                    df[col] = np.where(z > 3, df[col].median(), df[col])
                st.success("Outliers treated using Z-Score method")

            # -------------------- SHOW BOX PLOTS AFTER OUTLIER TREATMENT --------------------
            st.subheader("📦 Boxplot After Outlier Capping")

            cols_per_row = 2
            numeric_cols_after_capping = features  # Only show selected features

            for row in range((len(numeric_cols_after_capping) + cols_per_row - 1) // cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    var_idx = row * cols_per_row + col_idx
                    if var_idx < len(numeric_cols_after_capping):
                        variable = numeric_cols_after_capping[var_idx]

                        # Create capped box plot
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=df[variable],
                            name=f"{variable} (capped)",
                            boxmean='sd',
                            marker_color='lightgreen'
                        ))

                        fig.update_layout(
                            title=f'Box Plot After Capping: {variable}',
                            yaxis_title=variable,
                            height=400
                        )

                        cols[col_idx].plotly_chart(fig, use_container_width=True)
            small_footnote("After cleaning the outliers using either the Z-score or Interquartile Range method, the extreme values were capped, resulting in a dataset that appears more uniform and closer to a normal distribution.The Boxplots above displays the effects after capping the outliers, showing a more compact distribution with reduced variability.")

                           # -------------------- 3. FEATURE ENGINEERING --------------------
            st.subheader("⚙ Feature Engineering")

            if "date" in df.columns:
                df['year'] = pd.to_datetime(df['date']).dt.year
                df['month'] = pd.to_datetime(df['date']).dt.month
                df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week
                df['day'] = pd.to_datetime(df['date']).dt.day
                df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
                st.success("Date-based feature engineering applied")

            # Interaction features
            if st.checkbox("Create Interaction Features (Feature × Feature)"):
                for col1 in features:
                    for col2 in features:
                        if col1 != col2:
                            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                st.success("Interaction features created")

            # Polynomial features
            if st.checkbox("Add Polynomial Features (degree=2)"):
                from sklearn.preprocessing import PolynomialFeatures

                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(df[features])
                poly_df = pd.DataFrame(poly_features,
                                       columns=poly.get_feature_names_out(features))
                df = pd.concat([df, poly_df], axis=1)
                st.success("Polynomial features added")

            # -------------------- 4. FEATURE EXTRACTION --------------------
            st.subheader("🧬 Feature Extraction (Dimensionality Reduction)")

            use_pca = st.checkbox("Apply PCA (Principal Component Analysis)")

            if use_pca:
                from sklearn.decomposition import PCA

                n_components = st.slider("Select number of PCA Components", 1, len(features), 2)
                pca = PCA(n_components=n_components)
                pca_data = pca.fit_transform(df[features])
                df_pca = pd.DataFrame(pca_data, columns=[f'PC_{i + 1}' for i in range(n_components)])
                df = pd.concat([df, df_pca], axis=1)
                st.success(f"PCA Applied → {n_components} components extracted")

            # -------------------- 5. FEATURE SCALING --------------------
            st.subheader("📏 Feature Scaling")

            scaler_choice = st.radio(
                "Scaling Method",
                ["StandardScaler", "MinMaxScaler", "RobustScaler"]
            )

            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

            if scaler_choice == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_choice == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()

            df[features] = scaler.fit_transform(df[features])
            st.success(f"Features scaled using {scaler_choice}")
            st.write("Preview of Scaled Data:")
            st.dataframe(df[features].head())

            # ===== HOLD-OUT METHOD: BASELINE LINEAR REGRESSION FOR TRAIN vs TEST COMPARISON =====
            st.subheader("📊 Hold-Out Method: Train vs Test Performance")

            from sklearn.linear_model import LinearRegression

            # ================================
            # TRAIN–TEST SPLIT (REQUIRED)
            # ================================
            from sklearn.model_selection import train_test_split

            # X = features dataset, y = target variable
            X = df[features]
            y = df[target]  # Example: "Weekly_Sales"

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            st.success("Train-test split completed successfully!")
            st.write(f"Training samples: {X_train.shape[0]}")
            st.write(f"Testing samples: {X_test.shape[0]}")

            # Train baseline model
            holdout_model = LinearRegression()
            holdout_model.fit(X_train, y_train)

            # Predictions on train and test
            train_pred = holdout_model.predict(X_train)
            test_pred = holdout_model.predict(X_test)

            # Train metrics
            hold_train_mae = mean_absolute_error(y_train, train_pred)
            hold_train_mse = mean_squared_error(y_train, train_pred)
            hold_train_rmse = np.sqrt(hold_train_mse)
            hold_train_r2 = r2_score(y_train, train_pred)

            # Test metrics
            hold_test_mae = mean_absolute_error(y_test, test_pred)
            hold_test_mse = mean_squared_error(y_test, test_pred)
            hold_test_rmse = np.sqrt(hold_test_mse)
            hold_test_r2 = r2_score(y_test, test_pred)

            # Display results
            st.write("### 🟦 Train Set Performance (Hold-Out)")
            st.write(f"**MAE:** {hold_train_mae:.2f}")
            st.write(f"**MSE:** {hold_train_mse:.2f}")
            st.write(f"**RMSE:** {hold_train_rmse:.2f}")
            st.write(f"**R²:** {hold_train_r2:.4f}")

            st.write("### 🟥 Test Set Performance (Hold-Out)")
            st.write(f"**MAE:** {hold_test_mae:.2f}")
            st.write(f"**MSE:** {hold_test_mse:.2f}")
            st.write(f"**RMSE:** {hold_test_rmse:.2f}")
            st.write(f"**R²:** {hold_test_r2:.4f}")

            # ===== Detect Overfitting / Underfitting =====
            st.write("### 🔍 Model Fit Diagnosis (Hold-Out)")

            if hold_train_r2 > hold_test_r2 + 0.10:
                st.warning(
                    "⚠️ The model appears to be **overfitting** — high training accuracy but lower test accuracy.")
            elif hold_test_r2 > hold_train_r2 + 0.10:
                st.info("ℹ️ The model may be **underfitting** — performing better on test data than train data.")
            else:
                st.success("✅ The model shows **good generalization** — similar train and test performance.")

            # -------------------- 6. FEATURE SELECTION --------------------
            st.subheader("🎯 Feature Selection Techniques")

            selection_method = st.selectbox(
                "Choose Feature Selection Technique",
                ["None", "Correlation Filter", "SelectKBest (ANOVA)", "Lasso-based Selection"]
            )

            if selection_method == "Correlation Filter":
                corr_threshold = st.slider("Correlation threshold", 0.1, 0.9, 0.7)
                corr_matrix = df[features + [target]].corr()
                high_corr = corr_matrix[target].abs().sort_values(ascending=False)
                selected_features = high_corr[high_corr > corr_threshold].index.tolist()
                selected_features.remove(target)
                features = selected_features
                st.success(f"Selected features based on correlation: {features}")

            elif selection_method == "SelectKBest (ANOVA)":
                from sklearn.feature_selection import SelectKBest, f_regression

                k = st.slider("Select K Features", 1, len(features), 3)
                selector = SelectKBest(score_func=f_regression, k=k)
                selector.fit(df[features], df[target])
                selected = selector.get_support(indices=True)
                features = [features[i] for i in selected]
                st.success(f"Top {k} features selected: {features}")

            elif selection_method == "Lasso-based Selection":
                from sklearn.linear_model import Lasso

                lasso_selector = Lasso(alpha=0.01)
                lasso_selector.fit(df[features], df[target])
                selected = np.where(lasso_selector.coef_ != 0)[0]
                features = [features[i] for i in selected]
                st.success(f"Features selected using Lasso: {features}")

            # Final selected features
            st.info(f"🔎 Final Features Used for Modeling: {features}")

            # Slider to select test set size (10% to 40%)
            test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Multi-select widget to choose which models to train
            model_choice = st.multiselect(
                "Select Models to Train",
                ["Linear Regression", "Ridge", "Lasso", "Decision Tree",
                 "Random Forest", "Gradient Boosting", "SVR"],
                default=["Linear Regression", "Random Forest", "Gradient Boosting"]
            )

            # ===== NEW: HYPERPARAMETER TUNING OPTION =====
            st.subheader("⚙️ Model Configuration")
            use_hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning (slower but more accurate)",
                                                    value=False)
            use_cross_validation = st.checkbox("Enable Cross-Validation", value=True)

            if use_cross_validation:
                cv_folds = st.slider("Number of CV Folds", 3, 10, 5)  # Number of cross-validation folds

            # Button to trigger model training
            if st.button("🚀 Train Models"):
                models = {}  # Dictionary to store trained models
                results = []  # List to store performance metrics

                # Show spinner while training
                with st.spinner("Training models..."):

                    # ===== Train Linear Regression =====
                    if "Linear Regression" in model_choice:
                        lr = LinearRegression()  # Initialize model
                        lr.fit(X_train, y_train)  # Train on training data
                        models['Linear Regression'] = lr  # Store trained model

                    # ===== Train Ridge Regression with optional tuning =====
                    if "Ridge" in model_choice:
                        if use_hyperparameter_tuning:
                            # Define parameter grid for GridSearch
                            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
                            ridge = Ridge()
                            # GridSearchCV finds best alpha value
                            grid_search = GridSearchCV(ridge, param_grid, cv=cv_folds if use_cross_validation else 5,
                                                       scoring='r2', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            models['Ridge'] = grid_search.best_estimator_  # Store best model
                            st.info(f"Ridge best alpha: {grid_search.best_params_['alpha']}")
                        else:
                            ridge = Ridge(alpha=1.0)  # Default alpha value
                            ridge.fit(X_train, y_train)
                            models['Ridge'] = ridge

                    # ===== Train Lasso Regression with optional tuning =====
                    if "Lasso" in model_choice:
                        if use_hyperparameter_tuning:
                            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
                            lasso = Lasso()
                            grid_search = GridSearchCV(lasso, param_grid, cv=cv_folds if use_cross_validation else 5,
                                                       scoring='r2', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            models['Lasso'] = grid_search.best_estimator_
                            st.info(f"Lasso best alpha: {grid_search.best_params_['alpha']}")
                        else:
                            lasso = Lasso(alpha=1.0)
                            lasso.fit(X_train, y_train)
                            models['Lasso'] = lasso

                    # ===== Train Decision Tree with optional tuning =====
                    if "Decision Tree" in model_choice:
                        if use_hyperparameter_tuning:
                            # Tune max_depth and min_samples_split
                            param_grid = {
                                'max_depth': [3, 5, 7, 10],
                                'min_samples_split': [2, 5, 10]
                            }
                            dt = DecisionTreeRegressor(random_state=42)
                            grid_search = GridSearchCV(dt, param_grid, cv=cv_folds if use_cross_validation else 5,
                                                       scoring='r2', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            models['Decision Tree'] = grid_search.best_estimator_
                            st.info(f"Decision Tree best params: {grid_search.best_params_}")
                        else:
                            dt = DecisionTreeRegressor(random_state=42, max_depth=5)
                            dt.fit(X_train, y_train)
                            models['Decision Tree'] = dt

                    # ===== Train Random Forest with optional tuning =====
                    if "Random Forest" in model_choice:
                        if use_hyperparameter_tuning:
                            # Tune n_estimators and max_depth
                            param_grid = {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [5, 10, None],
                                'min_samples_split': [2, 5]
                            }
                            rf = RandomForestRegressor(random_state=42)
                            grid_search = GridSearchCV(rf, param_grid, cv=cv_folds if use_cross_validation else 5,
                                                       scoring='r2', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            models['Random Forest'] = grid_search.best_estimator_
                            st.info(f"Random Forest best params: {grid_search.best_params_}")
                        else:
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X_train, y_train)
                            models['Random Forest'] = rf

                    # ===== Train Gradient Boosting with optional tuning =====
                    if "Gradient Boosting" in model_choice:
                        if use_hyperparameter_tuning:
                            # Tune learning_rate and n_estimators
                            param_grid = {
                                'n_estimators': [50, 100, 200],
                                'learning_rate': [0.01, 0.1, 0.2],
                                'max_depth': [3, 5, 7]
                            }
                            gb = GradientBoostingRegressor(random_state=42)
                            grid_search = GridSearchCV(gb, param_grid, cv=cv_folds if use_cross_validation else 5,
                                                       scoring='r2', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            models['Gradient Boosting'] = grid_search.best_estimator_
                            st.info(f"Gradient Boosting best params: {grid_search.best_params_}")
                        else:
                            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
                            gb.fit(X_train, y_train)
                            models['Gradient Boosting'] = gb

                    # ===== Train SVR with optional tuning =====
                    if "SVR" in model_choice:
                        if use_hyperparameter_tuning:
                            # Tune C and gamma parameters
                            param_grid = {
                                'C': [0.1, 1, 10],
                                'gamma': ['scale', 'auto'],
                                'kernel': ['rbf']
                            }
                            svr = SVR()
                            grid_search = GridSearchCV(svr, param_grid, cv=cv_folds if use_cross_validation else 5,
                                                       scoring='r2', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            models['SVR'] = grid_search.best_estimator_
                            st.info(f"SVR best params: {grid_search.best_params_}")
                        else:
                            svr = SVR(kernel='rbf')
                            svr.fit(X_train, y_train)
                            models['SVR'] = svr

                    # ===== Evaluate each trained model =====
                    for name, model in models.items():
                        y_pred = model.predict(X_test)  # Make predictions on test set

                        # Calculate performance metrics
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
                        mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
                        r2 = r2_score(y_test, y_pred)  # R-squared score (0-1, higher is better)

                        # ===== NEW: CROSS-VALIDATION SCORE =====
                        if use_cross_validation:
                            # Perform k-fold cross-validation on entire dataset
                            cv_scores = cross_val_score(model, X, y, cv=cv_folds,
                                                        scoring='r2', n_jobs=-1)
                            cv_mean = cv_scores.mean()  # Average CV score
                            cv_std = cv_scores.std()  # Standard deviation of CV scores
                        else:
                            cv_mean = None
                            cv_std = None

                        # Store results in dictionary
                        results.append({
                            'Model': name,
                            'RMSE': round(rmse, 2),
                            'MAE': round(mae, 2),
                            'R² Score': round(r2, 4),
                            'CV Mean R²': round(cv_mean, 4) if cv_mean is not None else 'N/A',
                            'CV Std': round(cv_std, 4) if cv_std is not None else 'N/A'
                        })

                # Display training completion message
                st.success("✅ Training completed!")

                # Convert results list to DataFrame for display
                results_df = pd.DataFrame(results)
                st.subheader("📊 Model Performance Comparison")

                # Show information about cross-validation if enabled
                if use_cross_validation:
                    st.info(f"✓ Cross-validation performed with {cv_folds} folds. CV scores show model generalization.")

                # Create two columns for side-by-side display
                col1, col2 = st.columns(2)

                # Left column: Display results table with highlighting
                with col1:
                    # Highlight best scores (minimum RMSE/MAE, maximum R²)
                    st.dataframe(results_df.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                                 .highlight_max(subset=['R² Score'], color='lightgreen'),
                                 height=300)

                # Right column: Display bar chart of R² scores
                with col2:
                    fig = px.bar(results_df, x='Model', y='R² Score',
                                 title='R² Score Comparison',
                                 color='R² Score',
                                 color_continuous_scale='viridis')  # Color gradient
                    st.plotly_chart(fig, use_container_width=True)

                # ===== Predictions Visualization =====
                st.subheader("🎯 Predictions vs Actual")

                # Find the best performing model (highest R² score)
                best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
                best_model = models[best_model_name]  # Get the best model object
                y_pred_best = best_model.predict(X_test)  # Make predictions

                # Create line plot comparing actual vs predicted values
                fig = go.Figure()
                # Add actual values line
                fig.add_trace(go.Scatter(y=y_test.values, mode='lines+markers',
                                         name='Actual', line=dict(color='blue')))
                # Add predicted values line
                fig.add_trace(go.Scatter(y=y_pred_best, mode='lines+markers',
                                         name=f'Predicted ({best_model_name})',
                                         line=dict(color='red', dash='dash')))
                # Update plot layout
                fig.update_layout(title=f'Best Model: {best_model_name}',
                                  xaxis_title='Sample Index',
                                  yaxis_title=target)
                st.plotly_chart(fig, use_container_width=True)

                # ===== RESIDUAL ANALYSIS & HETEROSCEDASTICITY CHECK =====
                st.subheader("📉 Residual Analysis & Heteroscedasticity Check")

                # Calculate residuals (errors)
                residuals = y_test.values - y_pred_best

                col1, col2 = st.columns(2)

                with col1:
                    # Residuals vs Predicted Values (Check for Heteroscedasticity)
                    fig = px.scatter(x=y_pred_best, y=residuals,
                                     labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                     title='Residual Plot (Heteroscedasticity Check)')

                    # Add horizontal line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="red")

                    # Add interpretation text
                    fig.add_annotation(
                        text="Random scatter = Good<br>Pattern = Heteroscedasticity",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        bgcolor="lightyellow",
                        bordercolor="black",
                        borderwidth=1,
                        align="left"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Statistical test for heteroscedasticity (simple variance check)
                    # Split predictions into low and high groups
                    median_pred = np.median(y_pred_best)
                    low_group_residuals = residuals[y_pred_best < median_pred]
                    high_group_residuals = residuals[y_pred_best >= median_pred]

                    var_low = np.var(low_group_residuals)
                    var_high = np.var(high_group_residuals)
                    var_ratio = max(var_low, var_high) / min(var_low, var_high)

                    if var_ratio > 2:
                        st.warning(f"⚠️ Potential heteroscedasticity detected (variance ratio: {var_ratio:.2f})")
                        st.info("💡 Consider: data transformation, weighted regression, or robust standard errors")
                    else:
                        st.success(f"✅ Homoscedasticity assumption satisfied (variance ratio: {var_ratio:.2f})")

                with col2:
                    # Q-Q Plot (Check for Normality of Residuals)
                    from scipy import stats

                    # Generate theoretical quantiles
                    (theoretical_quantiles, ordered_residuals), _ = stats.probplot(residuals, dist="norm")

                    fig = go.Figure()
                    # Add scatter points
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=ordered_residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(color='blue')
                    ))

                    # Add reference line (perfect normal distribution)
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=theoretical_quantiles * np.std(residuals) + np.mean(residuals),
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', dash='dash')
                    ))

                    fig.update_layout(
                        title='Q-Q Plot (Normality Check)',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Shapiro-Wilk test for normality (for small samples)
                    if len(residuals) < 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                        if shapiro_p > 0.05:
                            st.success(f"✅ Residuals are normally distributed (p={shapiro_p:.4f})")
                        else:
                            st.warning(f"⚠️ Residuals may not be normal (p={shapiro_p:.4f})")
                    else:
                        st.info("Sample too large for Shapiro-Wilk test. Use Q-Q plot for visual assessment.")

                # Residuals Distribution Histogram
                st.subheader("📊 Residuals Distribution")
                fig = px.histogram(x=residuals, nbins=30,
                                   labels={'x': 'Residuals'},
                                   title='Distribution of Residuals')
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

                # Statistical summary of residuals
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                with col2:
                    st.metric("Std Dev", f"{np.std(residuals):.4f}")
                with col3:
                    st.metric("Min Residual", f"{np.min(residuals):.4f}")
                with col4:
                    st.metric("Max Residual", f"{np.max(residuals):.4f}")

                # ===== Feature Importance (only for tree-based models) =====
                if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                    st.subheader("📌 Feature Importance")

                    # Create DataFrame with feature names and their importance scores
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': best_model.feature_importances_  # Get importance from model
                    }).sort_values('Importance', ascending=False)  # Sort by importance

                    # Create horizontal bar chart
                    fig = px.bar(importance_df, x='Importance', y='Feature',
                                 orientation='h', title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
            small_footnote('Insights From The Baseline Model (Linear Regression)')
            small_footnote('Sales are highly non-linear with features like temperature, CPI, unemployment, and fuel price which could likely interact in non-linear ways that simple linear regression cannot capture.Linear regression performs poorly in this dataset.R² ~0.14 means only ~14% of the variance in weekly sales is explained by the features.MAE and RMSE are very high, indicating large errors in predictions.Train and test errors are almost identical meaning no overfitting, but clear underfitting(because the test set error metric is not greater than the train set).This suggests that the relationship between weekly sales and predictors like temperature, CPI, unemployment, holiday flags, and fuel prices is non-linear or more complex than linear regression can capture.')

            small_footnote('Insights From Using Ensemble Techniques Method')
            small_footnote('On overall performance,All tree-based models significantly outperform linear regression.R² values above 0.90 show that these models explain over 90% of the variance in weekly sales.RMSE and MAE values are drastically lower than linear regression, meaning predictions are much closer to actual sales.')
            small_footnote('Gradient Boosting: Best performance on both RMSE and MAE.Slightly lower CV R² than Random Forest, but within an acceptable range of (std = 0.0453).Gradient Boosting captures complex non-linear relationships and feature interactions better than the other models.')
            small_footnote('Random Forest has a very strong performance, second to Gradient Boosting.Slightly higher RMSE and MAE than Gradient Boosting.CV R² slightly higher than Gradient Boosting, which may indicate slightly better stability, but small differences are negligible. Whereas Decision Tree Performs worse than ensemble methods (Random Forest, Gradient Boosting).High risk of overfitting if tuned too aggressively, though CV indicates reasonable stability.')
            small_footnote('Recommendations and Conclusions')
            small_footnote('Ensemble models are more suitable: Random Forest and Gradient Boosting handle non-linearities and feature interactions better.Gradient Boosting is the best choice for this dataset: It has the lowest errors and high R², making it ideal for accurate weekly sales predictions.Consider hyperparameter tuning to further optimize performance (learning rate, number of estimators, max depth, etc.To sum up,Linear regression underfits and is unsuitable here. Tree-based ensemble models, especially Gradient Boosting, are highly effective for predicting weekly sales in this dataset due to their ability to model complex, non-linear relationships.')

        # ============ TIME SERIES ANALYSIS SECTION ============

        if analysis_type in ["Time Series Analysis", "Both"]:
            st.header("📈 Time Series Analysis")


        # --- Step 1: Select Date Column ---
        date_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
        if len(date_cols) > 0 or 'date' in df.columns:
            date_col = st.selectbox("Select Date Column", date_cols if len(date_cols) > 0 else ['date'])

            # --- Step 2: Parse Dates and Prepare Time Series ---
            ts_df = df[[date_col, target]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
            ts_df = ts_df.dropna(subset=[date_col, target])
            ts_df = ts_df.set_index(date_col).sort_index()
            ts_series = ts_df[target]

            # Plot original series
            st.subheader("📉 Original Time Series")
            fig = px.line(x=ts_series.index, y=ts_series.values, labels={'x': 'Date', 'y': target})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Footnote: The above plot shows the raw values of the target over time.")

            # --- Step 3: Seasonal Decomposition ---
            st.subheader("🔍 Seasonal Decomposition")
            if len(ts_series) >= 14:
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose

                    decomposition = seasonal_decompose(ts_series, model='additive', period=min(12, len(ts_series) // 2))

                    st.line_chart(pd.DataFrame({
                        'Trend': decomposition.trend,
                        'Seasonal': decomposition.seasonal,
                        'Residual': decomposition.resid
                    }))
                    st.caption(
                        "Footnote: Trend shows long-term direction, Seasonal captures repeating patterns, Residual shows noise.")
                except Exception as e:
                    st.warning(f"Decomposition failed: {str(e)}")
            else:
                st.warning("Series too short for meaningful decomposition.")

            # --- Step 4: Check Stationarity ---
            st.subheader("📊 Stationarity Check (ADF Test)")
            from statsmodels.tsa.stattools import adfuller

            adf_result = adfuller(ts_series)
            st.write(f"ADF Statistic: {adf_result[0]:.4f}")
            st.write(f"p-value: {adf_result[1]:.4f}")
            if adf_result[1] < 0.05:
                st.success("Series is stationary")
                ts_stationary = ts_series.copy()
            else:
                st.warning("Series is not stationary. Differencing applied.")
                ts_stationary = ts_series.diff().dropna()
            st.caption(
                "Footnote: Stationarity is required for ARIMA and some forecasting methods. Differencing removes trends if necessary.")

            # --- Step 5: Train/Test Split ---
            split_idx = int(len(ts_stationary) * 0.8)
            train, test = ts_stationary[:split_idx], ts_stationary[split_idx:]
            st.caption("Footnote: 80% of data used for training, 20% for testing model predictions.")

            # --- Step 6: Forecasting ---
            forecast_method = st.selectbox("Select Forecasting Method", ["Exponential Smoothing", "ARIMA"])
            forecast_periods = st.slider("Forecast Periods (Monthly Steps)", 12, 48, 48)  # 4 years ahead

            if st.button("📊 Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    try:
                        # Forecast
                        if forecast_method == "Exponential Smoothing":
                            from statsmodels.tsa.holtwinters import ExponentialSmoothing

                            model = ExponentialSmoothing(train, trend='add', seasonal=None)
                            fitted = model.fit()
                            forecast_values = fitted.forecast(len(test) + forecast_periods)
                        else:  # ARIMA
                            from pmdarima import auto_arima

                            auto_model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
                            fitted = auto_model
                            forecast_values = fitted.predict(n_periods=len(test) + forecast_periods)

                        # Forecast index
                        last_date = ts_series.index[-1]
                        forecast_index = pd.date_range(start=last_date, periods=len(test) + forecast_periods + 1,
                                                       freq='MS')[1:]

                        # --- Step 7: Plot Historical + Test + Forecast ---
                        st.subheader("📈 Test vs Predicted")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Test', line=dict(color='green')))
                        fig.add_trace(
                            go.Scatter(x=forecast_index[:len(test)], y=forecast_values[:len(test)], name='Predicted',
                                       line=dict(color='red', dash='dash')))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Footnote: Green line = actual test values; Red dashed line = model predictions on test set.")

                        # --- Step 8: Residual Analysis ---
                        residuals = test.values - forecast_values[:len(test)]
                        mse = np.mean(residuals ** 2)
                        rmse = np.sqrt(mse)
                        st.write(f"Residual Analysis - MSE: {mse:.4f}, RMSE: {rmse:.4f}")
                        st.caption(
                            "Footnote: Residuals measure prediction errors; lower RMSE/MSE indicates better fit.")

                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals'))
                        st.plotly_chart(fig_resid, use_container_width=True)
                        st.caption(
                            "Footnote: Residual plot helps check for patterns; ideally, residuals appear random.")

                        # --- Step 9: Forecast 4 Years Ahead ---
                        st.subheader("🔮 Forecast 4 Years Ahead")
                        future_index = pd.date_range(start=ts_series.index[-1], periods=forecast_periods + 1,
                                                     freq='MS')[1:]
                        future_forecast = forecast_values[-forecast_periods:]
                        future_df = pd.DataFrame({'Date': future_index, 'Forecasted Value': future_forecast})
                        st.dataframe(future_df)
                        st.caption("Footnote: Forecasted values for next 4 years (monthly steps).")

                        # --- Step 10: Plot 4-Year Forecast ---
                        st.subheader("📊 4-Year Forecast Plot")
                        fig_future = go.Figure()
                        fig_future.add_trace(go.Scatter(x=ts_series.index, y=ts_series.values, name='Historical'))
                        fig_future.add_trace(go.Scatter(x=future_index, y=future_forecast, name='Forecast',
                                                        line=dict(color='red', dash='dash')))
                        st.plotly_chart(fig_future, use_container_width=True)
                        st.caption("Footnote: Red dashed line shows predicted trend for the next 4 years.")

                    except Exception as e:
                        st.error(f"Forecasting failed: {str(e)}")
        else:
            st.warning("No date column found. Please provide a date column for time series analysis.")


# If no data is loaded, show instructions
else:
    # Display information message
    st.info("👈 Please upload a CSV file or generate sample data to begin analysis.")

    # Display helpful information about the app
    st.markdown("""
    ### 📝 Expected Data Format:
    - **For Regression**: Numeric columns for features and target variable
    - **For Time Series**: Date column + numeric target column

    ### 🎯 Features:
    - Multiple regression algorithms (Linear, Ridge, Random Forest, etc.)
    - Time series decomposition and forecasting
    - Interactive visualizations
    - Model performance comparison
    """)

# Footer section
st.markdown("---")  # Horizontal line separator

st.markdown("Built with Streamlit 🎈 | Sales Prediction Analytics")  # Footer text
