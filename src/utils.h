// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_map>

// Function to generate one-hot encodings and class names
std::tuple<std::unordered_map<std::string, torch::Tensor>, std::vector<std::string>>
generateOneHotEncoding(const std::unordered_map<std::string, std::vector<std::string>>& dataset);

// Function to perform stratified split of the dataset
std::tuple<std::vector<std::string>, std::vector<torch::Tensor>,
           std::vector<std::string>, std::vector<torch::Tensor>,
           std::vector<std::string>, std::vector<torch::Tensor>>
stratifiedSplit(const std::unordered_map<std::string, std::vector<std::string>>& dataset,
                const std::unordered_map<std::string, torch::Tensor>& one_hot_labels,
                float train_ratio,
                float val_ratio);

// Function to display an image with a label using OpenCV
void displayImageWithLabel(const torch::Tensor& img_tensor, const std::string& label_text);

#endif // UTILS_H
