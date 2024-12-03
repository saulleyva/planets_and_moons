#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_map>
#include "dataset.h"
#include "model.h"
#include "utils.h"


int main(int argc, char* argv[]) {
    float train_ratio = 0.7;
    float val_ratio = 0.15;
    std::string base_path = "C:\\Users\\sauls\\Desktop\\planets_and_moons\\data";
    std::string best_model_path = "best_model.pt";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train_ratio" && i + 1 < argc) {
            train_ratio = std::stof(argv[++i]);
        } else if (arg == "--val_ratio" && i + 1 < argc) {
            val_ratio = std::stof(argv[++i]);
        } else if (arg == "--base_path" && i + 1 < argc) {
            base_path = argv[++i];
        } else if (arg == "--best_model_path" && i + 1 < argc) {
            best_model_path = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }

    if (!std::filesystem::exists(base_path)) {
        std::cerr << "Error: Base path does not exist: " << base_path << std::endl;
        return 1;
    }

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    std::unordered_map<std::string, std::vector<std::string>> dataset;

    for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
        if (entry.is_directory()) {
            std::string planet = entry.path().filename().string();
            for (const auto& img_entry : std::filesystem::directory_iterator(entry.path())) {
                dataset[planet].push_back(img_entry.path().string());
            }
        }
    }

    auto [one_hot_labels, class_names] = generateOneHotEncoding(dataset);

    auto [train_set, train_labels, val_set, val_labels, test_set, test_labels] = stratifiedSplit(dataset, one_hot_labels, train_ratio, val_ratio);

    auto test_dataset = Dataset(test_set, test_labels).map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), 
                                                torch::data::DataLoaderOptions().batch_size(1).workers(1));

    auto model = std::make_shared<PlanetNet>(one_hot_labels.size());
    torch::load(model, best_model_path);
    model->to(device);
    model->eval(); 

    {
        torch::NoGradGuard no_grad;
        for (auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            int true_class_index = target.argmax(1).item<int>();

            auto output = model->forward(data);
            int predicted_class_index = output.argmax(1).item<int>();

            std::string true_class_name = class_names[true_class_index];
            std::string predicted_class_name = class_names[predicted_class_index];

            std::string label_text = "True: " + true_class_name + "\nPred: " + predicted_class_name;

            displayImageWithLabel(data.squeeze(0).cpu(), label_text);
        }
    }

    return 0;
}