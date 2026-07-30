# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "shubhamrangdal2000"
    repo_name = "MLOPS_CAPSTONE_Project"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()
    model_name = "my_model"

    def get_latest_version_for_stage(model_name, stage):
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            stage_versions = [v for v in versions if v.current_stage and v.current_stage.lower() == stage.lower()]
            if stage_versions:
                stage_versions.sort(key=lambda x: int(x.version), reverse=True)
                return stage_versions[0].version
        except Exception:
            pass

        try:
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            if latest_versions:
                return latest_versions[0].version
        except Exception:
            pass

        return None

    # Get the latest version in staging
    latest_version_staging = get_latest_version_for_stage(model_name, "Staging")
    if not latest_version_staging:
        raise RuntimeError(f"No model versions found in 'Staging' for '{model_name}'.")

    # Archive the current production model
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception:
        prod_versions = []
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()
