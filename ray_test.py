import ray
import torch
import time
import os

ray.init()

print("=== Cluster Ressourcen ===")
print(ray.cluster_resources())

@ray.remote(num_gpus=1)
def gpu_training():
    import torch, time, os

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Training läuft auf: {gpu_name}")

    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(100):
        x = torch.randn(512, 100, device=device)
        y = torch.randint(0, 10, (512,), device=device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/100 - Loss: {loss.item():.4f}")

    # Model speichern
    save_path = "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model gespeichert: {save_path}")

    elapsed = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1e9

    return {
        "gpu": gpu_name,
        "time": f"{elapsed:.2f}s",
        "vram": f"{vram:.2f} GB",
        "final_loss": f"{loss.item():.4f}"
    }

result = ray.get(gpu_training.remote())
print(f"\nGPU:        {result['gpu']}")
print(f"Zeit:       {result['time']}")
print(f"VRAM:       {result['vram']}")
print(f"Final Loss: {result['final_loss']}")
print("\nModel wurde gespeichert als: best_model.pth ✅")