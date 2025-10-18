#!/bin/bash
# TorchServe Setup and Deployment Script

echo "========================================"
echo "  TorchServe Deployment Setup"
echo "========================================"

# 1. Install TorchServe
echo "[Step 1] Installing TorchServe..."
pip install torchserve torch-model-archiver torch-workflow-archiver

# 2. Create directory structure
echo "[Step 2] Creating directories..."
mkdir -p model_store
mkdir -p logs

# 3. Create model archive
echo "[Step 3] Creating model archive..."

# Create handler
cat > handler.py << 'EOF'
import torch
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FashionRecommendationHandler:
    """TorchServe handler for real-time recommendations"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        self.user_embeddings = None
        self.item_embeddings = None
        self.item_ids = None
        self.mappings = None
    
    def initialize(self, context):
        """Load model and artifacts"""
        logger.info("Initializing handler...")
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model
        model_path = f"{model_dir}/fashion_graphsage.pt"
        
        from model import MultiModalGraphSAGE
        
        self.model = MultiModalGraphSAGE(num_users=50000, num_items=44000, embed_dim=128)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Precompute embeddings for fast inference
        logger.info("Precomputing embeddings...")
        with torch.no_grad():
            # Load graph structure (simplified for inference)
            edge_index = torch.load(f"{model_dir}/edge_index.pt")
            self.user_embeddings, self.item_embeddings = self.model(edge_index)
        
        # Load mappings
        with open(f"{model_dir}/mappings.json", 'r') as f:
            self.mappings = json.load(f)
        
        # Load item IDs
        embeddings_data = np.load(f"{model_dir}/embeddings.npz")
        self.item_ids = embeddings_data['item_ids']
        
        self.initialized = True
        logger.info("Handler initialized successfully")
    
    def preprocess(self, requests):
        """Preprocess incoming requests"""
        batch = []
        
        for request in requests:
            data = request.get("body") or request.get("data")
            
            if isinstance(data, (bytes, bytearray)):
                data = data.decode('utf-8')
            
            request_json = json.loads(data)
            batch.append(request_json)
        
        return batch
    
    def inference(self, batch):
        """Batch inference"""
        results = []
        
        for request in batch:
            user_id = request.get('user_id')
            k = request.get('k', 10)
            exclude_items = request.get('exclude_items', [])
            
            # Get user index
            user_idx = self.mappings['user_to_idx'].get(user_id)
            
            if user_idx is None:
                results.append({'error': 'User not found', 'user_id': user_id})
                continue
            
            # Get scores (fast: just dot product)
            user_emb = self.user_embeddings[user_idx]
            scores = torch.matmul(user_emb, self.item_embeddings.t()).cpu().numpy()
            
            # Exclude items
            for item_id in exclude_items:
                item_idx = self.mappings['item_to_idx'].get(item_id)
                if item_idx is not None:
                    scores[item_idx] = -np.inf
            
            # Get top-K
            top_k_indices = np.argsort(scores)[-k:][::-1]
            
            recommendations = [
                {
                    'item_id': str(self.item_ids[idx]),
                    'score': float(scores[idx]),
                    'rank': i + 1
                }
                for i, idx in enumerate(top_k_indices)
            ]
            
            results.append({
                'user_id': user_id,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    def postprocess(self, inference_output):
        """Format response"""
        return [json.dumps(output) for output in inference_output]


_service = FashionRecommendationHandler()


def handle(data, context):
    """Main entry point"""
    if not _service.initialized:
        _service.initialize(context)
    
    if data is None:
        return None
    
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    
    return data
EOF

echo "✅ Handler created"

# 4. Create model archiver script
cat > create_mar.sh << 'EOF'
#!/bin/bash
# Create TorchServe model archive

torch-model-archiver \
  --model-name fashion_rec \
  --version 1.0 \
  --model-file src/graphsage/model.py \
  --serialized-file models/best_fashion_graphsage.pt \
  --handler handler.py \
  --extra-files "models/embeddings.npz,models/mappings.json,models/edge_index.pt" \
  --export-path model_store/ \
  --force

echo "✅ Model archive created: model_store/fashion_rec.mar"
EOF

chmod +x create_mar.sh

# 5. Create TorchServe config
cat > config.properties << 'EOF'
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
number_of_gpu=1
max_request_size=104857600
max_response_size=104857600
default_workers_per_model=4
EOF

echo "✅ TorchServe config created"

# 6. Start TorchServe
echo "[Step 4] Starting TorchServe..."
cat > start_torchserve.sh << 'EOF'
#!/bin/bash

# Stop any existing TorchServe
torchserve --stop

# Start TorchServe
torchserve \
  --start \
  --model-store model_store \
  --models fashion_rec=fashion_rec.mar \
  --ts-config config.properties

# Wait for startup
sleep 10

# Register model
curl -X POST "http://localhost:8081/models?url=fashion_rec.mar&initial_workers=4&synchronous=true"

# Health check
echo ""
echo "Health check:"
curl http://localhost:8080/ping

echo ""
echo "✅ TorchServe started successfully"
echo "Inference endpoint: http://localhost:8080/predictions/fashion_rec"
echo "Management endpoint: http://localhost:8081"
echo "Metrics endpoint: http://localhost:8082/metrics"
EOF

chmod +x start_torchserve.sh

# 7. Create test inference script
cat > test_inference.py << 'EOF'
import requests
import json
import time

# Test request
request_data = {
    "user_id": "user_000123",
    "k": 10
}

print("Sending test request...")
start = time.time()

response = requests.post(
    "http://localhost:8080/predictions/fashion_rec",
    json=request_data
)

latency = (time.time() - start) * 1000

print(f"\nResponse (latency: {latency:.1f}ms):")
print(json.dumps(response.json(), indent=2))

# Benchmark
print("\nRunning benchmark (100 requests)...")
latencies = []

for i in range(100):
    start = time.time()
    response = requests.post(
        "http://localhost:8080/predictions/fashion_rec",
        json={"user_id": f"user_{i:06d}", "k": 10}
    )
    latencies.append((time.time() - start) * 1000)

print(f"\nBenchmark results:")
print(f"  Average: {np.mean(latencies):.1f}ms")
print(f"  P95: {np.percentile(latencies, 95):.1f}ms")
print(f"  P99: {np.percentile(latencies, 99):.1f}ms")
print(f"  Max: {np.max(latencies):.1f}ms")
EOF

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Create model archive: ./create_mar.sh"
echo "2. Start TorchServe: ./start_torchserve.sh"
echo "3. Test inference: python test_inference.py"
echo ""
echo "Then start production pipeline:"
echo "  python production_pipeline.py"