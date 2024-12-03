// dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>

class Dataset  : public torch::data::Dataset<Dataset > {
private:
    std::vector<std::string> image_paths;
    std::vector<torch::Tensor> labels;

public:
    Dataset (const std::vector<std::string>& images, const std::vector<torch::Tensor>& labels)
        : image_paths(images), labels(labels) {}

    torch::optional<size_t> size() const override {
        return image_paths.size();
    }

    torch::data::Example<> get(size_t index) override {
        cv::Mat img = cv::imread(image_paths[index], cv::IMREAD_COLOR);

        auto img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3},
                                                      torch::kUInt8).permute({2, 0, 1}).to(torch::kFloat32).div(255.0).clone();
        auto label_tensor = labels[index];

        return {img_tensor, label_tensor};
    }
};

#endif // DATASET_H
