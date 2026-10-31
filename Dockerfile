FROM python:3.13.15-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py dashboard.py fraud_core.py client.py ./
COPY config.json preprocessor.pkl xgboost_model.json versions.json ./
COPY reference.csv demo_transactions.csv dashboard_validation.csv global_shap_importance.csv test_summary.json ./
COPY .streamlit/config.toml ./.streamlit/config.toml
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
