from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import os
import random

app = Flask(__name__)

# Create static folder for graphs if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

def process_and_test_models():
    # 1. LOAD DATA
    csv_path = 'data set/global_cattle_disease_detection_dataset (1).csv'
    df_full = pd.read_csv(csv_path)
    # Use a sample for performance (10,000 rows)
    df = df_full.sample(n=10000, random_state=42).copy()
    
    # 2. PREPROCESSING
    df['Disease_Status'] = df['Disease_Status'].apply(lambda x: 'Healthy' if x == 'Healthy' else 'Diseased')
    
    categorical_cols = [
        'Breed', 'Region', 'Country', 'Climate_Zone', 'Management_System', 
        'Lactation_Stage', 'Feed_Type', 'Season'
    ]
    numeric_cols = [
        'Age_Months', 'Weight_kg', 'Parity', 'Days_in_Milk', 
        'Feed_Quantity_kg', 'Water_Intake_L', 'Walking_Distance_km', 
        'Grazing_Duration_hrs', 'Rumination_Time_hrs', 'Resting_Hours', 
        'Body_Temperature_C', 'Heart_Rate_bpm', 'Respiratory_Rate', 
        'Ambient_Temperature_C', 'Humidity_percent', 'Housing_Score', 
        'Milk_Yield_L', 'FMD_Vaccine', 'Brucellosis_Vaccine', 'HS_Vaccine', 
        'BQ_Vaccine', 'Anthrax_Vaccine', 'IBR_Vaccine', 'BVD_Vaccine', 
        'Rabies_Vaccine', 'Previous_Week_Avg_Yield', 'Body_Condition_Score', 
        'Milking_Interval_hrs'
    ]
    
    selected_features = categorical_cols + numeric_cols
    target = 'Disease_Status'
    
    df = df[selected_features + [target]].dropna()
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    le_target = LabelEncoder()
    df_encoded[target] = le_target.fit_transform(df[target])
    encoders[target] = le_target
        
    X = df_encoded[selected_features]
    y = df_encoded[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features for KNN
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 3. MODELS
    # KNN (using scaled data)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_prec = precision_score(y_test, knn_pred)
    knn_rec = recall_score(y_test, knn_pred)
    knn_f1 = f1_score(y_test, knn_pred)
    knn_cm = confusion_matrix(y_test, knn_pred)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_prec = precision_score(y_test, dt_pred)
    dt_rec = recall_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    dt_cm = confusion_matrix(y_test, dt_pred)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    # 4. GRAPHS
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Disease_Status', data=df, palette='viridis')
    plt.title('Disease Status Distribution (Sampled)')
    plt.savefig('static/countplot.png')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('static/heatmap.png')
    plt.close()
    
    k_values = range(1, 11)
    k_acc = []
    for k in k_values:
        temp_knn = KNeighborsClassifier(n_neighbors=k)
        temp_knn.fit(X_train_scaled, y_train)
        k_acc.append(accuracy_score(y_test, temp_knn.predict(X_test_scaled)))
    
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, k_acc, marker='o', color='teal')
    plt.title('K vs Accuracy for KNN')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('static/k_vs_acc.png')
    plt.close()
    
    models = ['KNN', 'Decision Tree', 'Random Forest']
    accuracies = [knn_acc, dt_acc, rf_acc]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=models, y=accuracies, palette='magma')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig('static/model_comparison.png')
    plt.close()

    # 5. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cms = [knn_cm, dt_cm, rf_cm]
    for i, (cm, name) in enumerate(zip(cms, models)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/confusion_matrices.png')
    plt.close()

    # 6. Feature Importance (RF & DT)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    
    # RF Importance (top 20)
    rf_importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=True).tail(20)
    rf_importances.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title('Random Forest Top 20 Feature Importance')
    
    # DT Importance (top 20)
    dt_importances = pd.Series(dt.feature_importances_, index=selected_features).sort_values(ascending=True).tail(20)
    dt_importances.plot(kind='barh', ax=ax2, color='salmon')
    ax2.set_title('Decision Tree Top 20 Feature Importance')
    
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()

    # 7. Multi-metric Comparison
    metrics_df = pd.DataFrame({
        'Model': models * 3,
        'Score': [knn_prec, dt_prec, rf_prec, knn_rec, dt_rec, rf_rec, knn_f1, dt_f1, rf_f1],
        'Metric': ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='Set2')
    plt.title('Model Performance: Precision, Recall, F1-Score')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('static/multi_metric_comparison.png')
    plt.close()

    # 8. ROC Curves
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8, 6))
    
    # KNN ROC
    knn_probs = knn.predict_proba(X_test_scaled)[:, 1]
    fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)
    auc_knn = auc(fpr_knn, tpr_knn)
    plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})', color='teal')
    
    # DT ROC
    dt_probs = dt.predict_proba(X_test)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
    auc_dt = auc(fpr_dt, tpr_dt)
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='salmon')
    
    # RF ROC
    rf_probs = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='skyblue')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/roc_curves.png')
    plt.close()

    dropdown_vals = {col: encoders[col].classes_.tolist() for col in categorical_cols}

    return {
        'df_sample': df.head(10).to_html(classes='table table-striped', index=False),
        'accuracies': {'KNN': knn_acc, 'Decision Tree': dt_acc, 'Random Forest': rf_acc},
        'precisions': {'KNN': knn_prec, 'Decision Tree': dt_prec, 'Random Forest': rf_prec},
        'recalls': {'KNN': knn_rec, 'Decision Tree': dt_rec, 'Random Forest': rf_rec},
        'f1_scores': {'KNN': knn_f1, 'Decision Tree': dt_f1, 'Random Forest': rf_f1},
        'aucs': {'KNN': auc_knn, 'Decision Tree': auc_dt, 'Random Forest': auc_rf},
        'cms': {'KNN': knn_cm.tolist(), 'Decision Tree': dt_cm.tolist(), 'Random Forest': rf_cm.tolist()},
        'rf_model': rf,
        'knn_model': knn,
        'dt_model': dt,
        'scaler': scaler,
        'encoders': encoders,
        'selected_features': selected_features,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'dropdown_vals': dropdown_vals,
        'raw_df': df_full[selected_features] 
    }

results = process_and_test_models()

@app.route('/')
def index():
    return render_template('index.html', data=results)

@app.route('/comparison')
def comparison():
    return render_template('comparison.html', data=results)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html', data=results)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_inputs = []
        for col in results['selected_features']:
            val = request.form.get(col)
            if col in results['categorical_cols']:
                le = results['encoders'][col]
                try:
                    encoded_val = le.transform([val])[0]
                except:
                    encoded_val = 0
                raw_inputs.append(encoded_val)
            else:
                raw_inputs.append(float(val))

        input_df = pd.DataFrame([raw_inputs], columns=results['selected_features'])
        # Ensure all columns are numeric for the model
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Scale for KNN if needed, but here we use RF for main prediction as before
        rf = results['rf_model']
        prediction = rf.predict(input_df)[0]
        
        le_target = results['encoders']['Disease_Status']
        result = le_target.inverse_transform([prediction])[0]
        
        return render_template('prediction.html', data=results, prediction_text=f'Result: {result}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('prediction.html', data=results, prediction_text=f'Error: {str(e)}')

@app.route('/get_random_data')
def get_random_data():
    random_row = results['raw_df'].sample(1).iloc[0].to_dict()
    # Convert all values to string to avoid JSON serializable issues with numpy types
    for key in random_row:
        random_row[key] = str(random_row[key])
    return jsonify(random_row)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
