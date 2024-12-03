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
    int batch_size = 8;
    int num_workers = 2;
    int num_epochs = 10;
    float learning_rate = 1e-4;
    std::string base_path = "C:\\Users\\sauls\\Desktop\\planets_and_moons\\data";
    std::string best_model_path = "best_model.pt";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train_ratio" && i + 1 < argc) {
            train_ratio = std::stof(argv[++i]);
        } else if (arg == "--val_ratio" && i + 1 < argc) {
            val_ratio = std::stof(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--num_workers" && i + 1 < argc) {
            num_workers = std::stoi(argv[++i]);
        } else if (arg == "--num_epochs" && i + 1 < argc) {
            num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--learning_rate" && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
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

    std::cout << "Training set size: " << train_set.size() << ", Training labels size: " << train_labels.size() << std::endl;
    std::cout << "Validation set size: " << val_set.size() << ", Validation labels size: " << val_labels.size() << std::endl;
    std::cout << "Test set size: " << test_set.size() << ", Test labels size: " << test_labels.size() << std::endl;

    auto train_dataset = Dataset(train_set, train_labels).map(torch::data::transforms::Stack<>());
    auto val_dataset = Dataset(val_set, val_labels).map(torch::data::transforms::Stack<>());
    auto test_dataset = Dataset(test_set, test_labels).map(torch::data::transforms::Stack<>());

    int total_train_batches = (train_dataset.size().value() + batch_size - 1) / batch_size;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), 
                                                torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(val_dataset), 
                                                torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset), 
                                                torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));

    float best_val_accuracy = 0.0;

    auto model = std::make_shared<PlanetNet>(one_hot_labels.size());
    model->to(device);

    auto criterion = torch::nn::NLLLoss();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        model->train();
        size_t batch_idx = 0;
        float running_loss = 0.0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            target = target.argmax(1);

            optimizer.zero_grad();

            auto output = model->forward(data);

            auto loss = criterion(output, target);

            loss.backward();
            optimizer.step();

            running_loss += loss.item<float>();

            if (++batch_idx % 10 == 0) {
                std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Step [" << batch_idx
                          << "/" << total_train_batches << "], Loss: " << running_loss / 10 << std::endl;
                running_loss = 0.0;
            }
        }

        // Validation loop
        model->eval();
        size_t correct = 0;
        size_t total = 0;
        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *val_loader) {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                target = target.argmax(1);

                auto outputs = model->forward(data);
                auto predicted = outputs.argmax(1);

                total += target.size(0);
                correct += (predicted == target).sum().item<int64_t>();
            }
        }

        float val_accuracy = static_cast<float>(correct) / total * 100.0;
        std::cout << "Validation Accuracy after epoch " << epoch << ": "
                  << val_accuracy << "%" << std::endl;
        
        if (val_accuracy > best_val_accuracy) {
            best_val_accuracy = val_accuracy;
            torch::save(model, best_model_path);
            std::cout << "New best model saved with validation accuracy: " << best_val_accuracy << "%" << std::endl;
        }
    }

    // Load the best model
    torch::load(model, best_model_path);
    model->to(device);
    std::cout << "Best model loaded for testing." << std::endl;

    // Testing loop
    model->eval();
    size_t correct = 0;
    size_t total = 0;
    {
        torch::NoGradGuard no_grad;
        for (auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            target = target.argmax(1);

            auto outputs = model->forward(data);
            auto predicted = outputs.argmax(1);

            total += target.size(0);
            correct += (predicted == target).sum().item<int64_t>();
        }
    }

    float test_accuracy = static_cast<float>(correct) / total * 100.0;
    std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;

    return 0;
}