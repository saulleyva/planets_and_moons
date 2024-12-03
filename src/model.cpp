// model.cpp
#include "model.h"

PlanetNet::PlanetNet(int num_classes) {
    conv1 = register_module("conv1", torch::nn::Conv2d(3, 16, 3));
    conv2 = register_module("conv2", torch::nn::Conv2d(16, 32, 3));
    fc1 = register_module("fc1", torch::nn::Linear(32 * 62 * 62, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, num_classes));
}

torch::Tensor PlanetNet::forward(torch::Tensor x) {
    x = torch::relu(conv1->forward(x));
    x = torch::max_pool2d(x, 2);
    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x, 2);
    x = x.view({-1, 32 * 62 * 62});
    x = torch::relu(fc1->forward(x));
    x = torch::log_softmax(fc2->forward(x), /*dim=*/1);
    return x;
}
