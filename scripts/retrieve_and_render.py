import os
import sys
import asyncio
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{str(PROJECT_ROOT / 'mlflow.db')}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application Details Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #8b5cf6;
            --accent-glow: rgba(139, 92, 246, 0.5);
        }
        
        body {
            margin: 0;
            padding: 0;
            min-height: 100vh;
            font-family: 'Inter', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container {
            width: 90%;
            max-width: 1000px;
            padding: 2rem;
            position: relative;
        }

        .container::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 60%;
            height: 60%;
            background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
            filter: blur(100px);
            z-index: -1;
            animation: pulse 4s ease-in-out infinite alternate;
        }

        @keyframes pulse {
            0% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
            100% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.1); }
        }

        header {
            text-align: center;
            margin-bottom: 4rem;
            animation: fadeInDown 1s ease-out;
        }

        h1 {
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(to right, #c4b5fd, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
        }

        .card {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.8s ease-out backwards;
        }

        .card:nth-child(1) { animation-delay: 0.1s; }
        .card:nth-child(2) { animation-delay: 0.2s; }
        .card:nth-child(3) { animation-delay: 0.3s; }
        .card:nth-child(4) { animation-delay: 0.4s; }

        .card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 20px var(--accent-glow);
        }

        .card-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            display: inline-block;
        }

        .card-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .card-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
            word-wrap: break-word;
        }
        
        .card-value.highlight {
            color: #34d399;
            text-shadow: 0 0 10px rgba(52, 211, 153, 0.3);
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Application State</h1>
            <div class="subtitle">Live System Telemetry & Configuration</div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-icon">🌊</div>
                <div class="card-title">Active Prefect Flow</div>
                <div class="card-value">{flow_name}</div>
            </div>
            
            <div class="card">
                <div class="card-icon">🧪</div>
                <div class="card-title">MLflow Experiment</div>
                <div class="card-value">{exp_name}</div>
            </div>
            
            <div class="card">
                <div class="card-icon">🤖</div>
                <div class="card-title">Registered Model</div>
                <div class="card-value">{model_name}</div>
            </div>
            
            <div class="card">
                <div class="card-icon">🚀</div>
                <div class="card-title">Deployment Stage</div>
                <div class="card-value highlight">{stage}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

async def get_prefect_flow():
    try:
        from prefect.client.orchestration import get_client
        async with get_client() as client:
            flows = await client.read_flows(limit=1)
            if flows:
                return flows[0].name
    except Exception as e:
        print(f"Warning: Could not fetch Prefect Flow ({e})")
    
    # Fallback to scanning the file directly if the server isn't running
    try:
        with open(PROJECT_ROOT / "scripts" / "orchestration" / "recomart_pipeline.py", "r") as f:
            for line in f:
                if "@flow(" in line and "name=" in line:
                    return line.split('name="')[1].split('"')[0]
    except Exception:
        pass
        
    return "RecoMart Pipeline" # Default

def get_mlflow_details():
    client = MlflowClient()
    
    exp_name = "Not Available"
    experiment = client.get_experiment_by_name("RecoMart_Experiments")
    if experiment:
        exp_name = experiment.name
        
    model_name = "Not Available"
    stage = "Not Available"
    
    try:
        registered_models = client.search_registered_models()
        if registered_models:
            model = registered_models[0]
            model_name = model.name
            if model.latest_versions:
                stage = model.latest_versions[0].current_stage
    except Exception as e:
        print(f"Error fetching MLflow: {e}")
        
    return exp_name, model_name, stage

async def main():
    print("Retrieving application details...")
    flow_name = await get_prefect_flow()
    exp_name, model_name, stage = get_mlflow_details()
    
    print(f"Flow: {flow_name}")
    print(f"Experiment: {exp_name}")
    print(f"Model: {model_name}")
    print(f"Stage: {stage}")
    
    html_content = HTML_TEMPLATE.replace(
        "{flow_name}", flow_name
    ).replace(
        "{exp_name}", exp_name
    ).replace(
        "{model_name}", model_name
    ).replace(
        "{stage}", stage
    )
    
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    output_path = docs_dir / "app_details.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Successfully generated HTML page at: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
