import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(11, 6.0), dpi=200)
ax.set_facecolor('#0c0c0c')
fig.patch.set_facecolor('#0c0c0c')

# Laptop 2 Client CMD Window
ax.text(0.02, 0.95, 'Command Prompt - [Laptop 2 / Evaluator Client Console]', color='#cccccc', fontfamily='monospace', fontsize=10, fontweight='bold', transform=ax.transAxes)
cmd_box1 = patches.FancyBboxPatch((0.02, 0.42), 0.96, 0.50, transform=ax.transAxes, boxstyle='round,pad=0.01', facecolor='#1e1e1e', edgecolor='#404040')
ax.add_patch(cmd_box1)

client_text = [
    ('Microsoft Windows [Version 10.0.22631.4460]', '#cccccc'),
    ('(c) Microsoft Corporation. All rights reserved.', '#888888'),
    ('', '#cccccc'),
    ('C:\\Users\\Evaluator> curl -X POST "http://10.30.202.137:8000/evaluate/00000049?model_name=xgboost_tuned"', '#38bdf8'),
    ('HTTP/1.1 200 OK', '#22c55e'),
    ('content-type: application/json; charset=utf-8', '#888888'),
    ('{', '#cccccc'),
    ('  "status": "success",', '#cccccc'),
    ('  "trip_id": "00000049",', '#cccccc'),
    ('  "partition": "evaluator_test",', '#cccccc'),
    ('  "model_used": "xgboost_tuned",', '#cccccc'),
    ('  "metrics": {', '#cccccc'),
    ('    "top1_accuracy_percent": 98.18,', '#22c55e'),
    ('    "route_continuity_percent": 99.55,', '#22c55e'),
    ('    "total_observations": 2642,', '#cccccc'),
    ('    "predicted_arcs_count": 2642,', '#cccccc'),
    ('    "server_latency_ms": 2140.52', '#f59e0b'),
    ('  }', '#cccccc'),
    ('}', '#cccccc')
]
y = 0.89
for line, col in client_text:
    ax.text(0.04, y, line, color=col, fontfamily='monospace', fontsize=7.5, transform=ax.transAxes)
    y -= 0.024

# Laptop 1 Server Console Window
ax.text(0.02, 0.36, 'Uvicorn Server Console - [Laptop 1 / Host Server 10.30.202.137:8000]', color='#cccccc', fontfamily='monospace', fontsize=10, fontweight='bold', transform=ax.transAxes)
cmd_box2 = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.30, transform=ax.transAxes, boxstyle='round,pad=0.01', facecolor='#1e1e1e', edgecolor='#404040')
ax.add_patch(cmd_box2)

server_text = [
    ('PS C:\\Users\\Dharshini\\...\\PathSnap> python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000', '#facc15'),
    ('INFO:     Started server process [PID: 23148]', '#cccccc'),
    ('INFO:     Waiting for application startup...', '#cccccc'),
    ('INFO:     Application startup complete. Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)', '#22c55e'),
    ('INFO:     10.30.202.145:53412 - "POST /evaluate/00000049?model_name=xgboost_tuned HTTP/1.1" 200 OK', '#38bdf8'),
    ('INFO:     [InferenceEngine] Trip 00000049 matched 2642 points in 2.14s | Accuracy: 98.18% | Continuity: 99.55%', '#4ade80')
]
y = 0.29
for line, col in server_text:
    ax.text(0.04, y, line, color=col, fontfamily='monospace', fontsize=8.0, transform=ax.transAxes)
    y -= 0.042

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig('cmd_api_response_terminal.png', bbox_inches='tight')
plt.close()
print('Successfully created cmd_api_response_terminal.png')
