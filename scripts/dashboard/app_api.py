import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.tracking import MlflowClient
from prefect.client.orchestration import get_client
app = FastAPI(title="RetailX Details API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
mlflow.set_tracking_uri(f"sqlite:///{str(PROJECT_ROOT / 'mlflow.db')}")

async def get_prefect_details():
    flow_runs_history = []
    prefect_deployments = []
    flow_name = "Not Available"
    
    try:
        
        async with get_client() as client:
            flows = await client.read_flows(limit=1)
            if flows:
                flow_name = flows[0].name
                
            runs = await client.read_flow_runs(limit=5)
            for r in runs:
                msg = r.state.message if r.state and r.state.message else ""
                if not msg and r.state:
                    msg = r.state.name
                    
                duration = "Running"
                if r.start_time and r.end_time:
                    delta = int((r.end_time - r.start_time).total_seconds())
                    duration = f"{delta}s"
                    
                flow_runs_history.append({
                    "id": str(r.id),
                    "name": r.name,
                    "state": r.state.name if r.state else "Unknown",
                    "start_time": r.start_time.strftime("%Y-%m-%d %H:%M:%S") if r.start_time else "N/A",
                    "duration": duration,
                    "message": msg
                })
                
            deps = await client.read_deployments(limit=5)
            for d in deps:
                schedule_str = "None"
                if hasattr(d, 'schedules') and d.schedules:
                    schedule_str = "Scheduled" # Simply indicate it has schedules
                elif hasattr(d, 'schedule') and d.schedule and hasattr(d.schedule, 'cron'):
                    schedule_str = d.schedule.cron
                    
                prefect_deployments.append({
                    "id": str(d.id),
                    "name": d.name,
                    "schedule": schedule_str,
                    "tags": ", ".join(d.tags) if d.tags else "None"
                })
    except Exception as e:
        print(f"Warning: Could not fetch Prefect Flow via API ({e})")
    
    if flow_name == "Not Available":
        # Fallback for demonstration if server isn't running properly
        try:
            with open(PROJECT_ROOT / "scripts" / "orchestration" / "retailx_pipeline.py", "r") as f:
                for line in f:
                    if "@flow(" in line and "name=" in line:
                        flow_name = line.split('name="')[1].split('"')[0]
        except Exception:
            pass
            
    return flow_name, flow_runs_history, prefect_deployments

def get_mlflow_details():
    client = MlflowClient()
    
    exp_name = "Not Available"
    experiment = client.get_experiment_by_name("RetailX_Experiments")
    
    mlflow_runs_history = []
    if experiment:
        exp_name = experiment.name
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=5,
                order_by=["start_time DESC"]
            )
            for r in runs:
                c_hr = r.data.metrics.get("content_hit_rate_50", "N/A")
                s_hr = r.data.metrics.get("svd_hit_rate_50", "N/A")
                c_pr = r.data.metrics.get("content_precision_50", "N/A")
                s_pr = r.data.metrics.get("svd_precision_50", "N/A")
                
                if c_hr != "N/A": c_hr = f"{c_hr:.4f}"
                if s_hr != "N/A": s_hr = f"{s_hr:.4f}"
                if c_pr != "N/A": c_pr = f"{c_pr:.4f}"
                if s_pr != "N/A": s_pr = f"{s_pr:.4f}"
                
                start = "N/A"
                if r.info.start_time:
                    import datetime
                    start = datetime.datetime.fromtimestamp(r.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

                model_type = r.data.params.get("model_type", "Unknown")

                mlflow_runs_history.append({
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "model_type": model_type,
                    "svd_hit_rate": s_hr,
                    "cb_hit_rate": c_hr,
                    "svd_precision": s_pr,
                    "cb_precision": c_pr,
                    "start_time": start
                })
        except Exception as e:
            print(f"Error fetching runs: {e}")
        
    model_name = "Not Available"
    stage = "Not Available"
    
    try:
        registered_models = client.search_registered_models()
        if registered_models:
            model = registered_models[0]
            model_name = model.name
            if model.latest_versions:
                stages = [v.current_stage for v in model.latest_versions]
                if "Production" in stages:
                    stage = "Production"
                elif "Staging" in stages:
                    stage = "Staging"
                else:
                    stage = stages[0] if stages else "None"
    except Exception as e:
        print(f"Error fetching MLflow models: {e}")
        
    return exp_name, model_name, stage, mlflow_runs_history

@app.get("/api/details")
async def get_details():
    flow_name, prefect_history, prefect_deployments = await get_prefect_details()
    exp_name, model_name, stage, mlflow_history = get_mlflow_details()
    
    return JSONResponse(content={
        "summary": {
            "flow_name": flow_name,
            "exp_name": exp_name,
            "model_name": model_name,
            "stage": stage
        },
        "history": {
            "prefect_runs": prefect_history,
            "prefect_deployments": prefect_deployments,
            "mlflow_runs": mlflow_history
        }
    })

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = PROJECT_ROOT /"scripts"/ "dashboard" / "app_details.html"
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard HTML not found. Please ensure docs/app_details.html exists.</h1>"
