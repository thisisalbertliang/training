import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
from torch import nn


class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    with xp.Trace('conv1'):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = self.bn1(x)
    with xp.Trace('conv2'):
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = self.bn2(x)
    with xp.Trace('dense'):
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def main():
    port = 8001
    server = xp.start_server(port)

    device = xm.xla_device()
    X = torch.rand(28 * 28, device=device)
    # model = nn.Sequential(
    #     # nn.Flatten(),
    #     nn.Linear(28 * 28, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 10),
    # )
    # model = model.to(device)

    with xp.StepTrace("train_dummy"):
        with xp.Trace("build_graph"):
            # while True:
            X = X + X
            # print("Finish X + X")
            # logits = model(X)
        xm.mark_step()
        # print("Finish `xm.mark_step()`")

    # xp.trace(
    #     service_addr=f"localhost:{port}",
    #     logdir="dummy_trace_logdir",
    #     duration_ms=3000,
    # )


if __name__ == "__main__":
    main()
