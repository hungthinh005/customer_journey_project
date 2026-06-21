-- Create the additional databases used by Airflow and MLflow.
-- The primary app database (POSTGRES_DB) is created automatically by the
-- postgres entrypoint. These run on first container init.

SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

GRANT ALL PRIVILEGES ON DATABASE airflow TO cjp;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO cjp;
