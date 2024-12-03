// model.h
#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>

struct PlanetNet : torch::nn::Module {
    PlanetNet(int num_classes);

    torch::Tensor forward(torch::Tensor x);

    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

#endif // MODEL_H
