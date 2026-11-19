# VI WebSync inference and training

This repository owns the private inference HTTP service, AI function interfaces,
and factory edge code. It includes the existing Ollama demo adapter. Dedicated
model training and packaging remain planned.

```bash
docker build -t vi-websync/inference:staging .
```

The backend stack runs this image on its private Docker network. It can reach
Ollama there when the vision profile is enabled. Review the contract at
packages/schemas/openapi/inference.json when changing inference endpoints.
