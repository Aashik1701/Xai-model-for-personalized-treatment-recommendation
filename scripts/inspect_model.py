from mlflow.tracking import MlflowClient

c = MlflowClient(tracking_uri="http://127.0.0.1:5000")
models = c.search_registered_models()
for m in models:
    if m.name == "BreastCancer":
        print("Model:", m.name)
        for v in m.latest_versions:
            print(
                "  Version:", v.version, "stage:", v.current_stage, "run_id:", v.run_id
            )
            try:
                mv = c.get_model_version(m.name, v.version)
                print("   source:", mv.source)
                # try to read MLmodel file inside the model version source if available
            except Exception as e:
                print("   error inspecting version:", e)
