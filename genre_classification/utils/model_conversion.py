import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.models as models

from genre_classification.utils.save_load import load_model


model_path = 'model_checkpoints/resnet34_checkpoint.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(model_path=model_path, model=models.resnet34(), device=device)

model.eval()

# Input to the model
batch_size = 1
frames = 10
x = torch.randn(batch_size, frames, 96)
embedding = model(x)

# Export the model
torch.onnx.export(
    model=model,
    args=x,
    f="resnet34.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 1: 'frames'},
                  'output': {0: 'batch_size'}}
)

onnx_model = onnx.load("resnet34.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("resnet34.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(embedding), ort_outs[0], rtol=1e-06, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

a = np.random.rand(64, 100, 96).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: a}
ort_outs = ort_session.run(None, ort_inputs)

print()