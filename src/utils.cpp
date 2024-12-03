// utils.cpp
#include "utils.h"

std::tuple<std::unordered_map<std::string, torch::Tensor>, std::vector<std::string>>
                                                generateOneHotEncoding(const std::unordered_map<std::string, 
                                                                       std::vector<std::string>>& dataset) {
    std::unordered_map<std::string, torch::Tensor> one_hot_labels;
    std::vector<std::string> class_names;
    int num_classes = dataset.size();
    int class_index = 0;

    for (const auto& [planet, _] : dataset) {
        torch::Tensor one_hot = torch::zeros({num_classes});
        one_hot[class_index] = 1;
        one_hot_labels[planet] = one_hot;
        class_names.push_back(planet);
        class_index++;
    }

    std::cout << "Generated One-Hot Encodings:\n";
    for (const auto& [planet, encoding] : one_hot_labels) {
        std::cout << planet << ": [";
        auto encoding_data = encoding.data_ptr<float>();
        for (int i = 0; i < encoding.size(0); ++i) {
            std::cout << encoding_data[i];
            if (i < encoding.size(0) - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
    return {one_hot_labels, class_names};
}

std::tuple<std::vector<std::string>, std::vector<torch::Tensor>, 
           std::vector<std::string>, std::vector<torch::Tensor>, 
           std::vector<std::string>, std::vector<torch::Tensor>> stratifiedSplit(const std::unordered_map<std::string, std::vector<std::string>>& dataset,
                                                                                 const std::unordered_map<std::string, torch::Tensor>& one_hot_labels,
                                                                                 float train_ratio, 
                                                                                 float val_ratio) {
    std::vector<std::string> train_set, val_set, test_set;
    std::vector<torch::Tensor> train_labels, val_labels, test_labels;
    unsigned int seed = 42;
    std::mt19937 gen(seed);

    for (const auto& [label, images] : dataset) {
        std::vector<std::string> shuffled_images = images;
        std::shuffle(shuffled_images.begin(), shuffled_images.end(), gen);
        
        int train_size = static_cast<int>(train_ratio * shuffled_images.size());
        int val_size = static_cast<int>(val_ratio * shuffled_images.size());

        train_set.insert(train_set.end(), shuffled_images.begin(), shuffled_images.begin() + train_size);
        train_labels.insert(train_labels.end(), train_size, one_hot_labels.at(label));

        val_set.insert(val_set.end(), shuffled_images.begin() + train_size, shuffled_images.begin() + train_size + val_size);
        val_labels.insert(val_labels.end(), val_size, one_hot_labels.at(label));

        test_set.insert(test_set.end(), shuffled_images.begin() + train_size + val_size, shuffled_images.end());
        test_labels.insert(test_labels.end(), shuffled_images.size() - train_size - val_size, one_hot_labels.at(label));
    }
    return {train_set, train_labels, val_set, val_labels, test_set, test_labels};
}


void displayImageWithLabel(const torch::Tensor& img_tensor, const std::string& label_text) {
    auto img = img_tensor.permute({1, 2, 0}).mul(255).clamp(0, 255).to(torch::kU8).contiguous();

    cv::Mat img_mat(cv::Size(img.size(1), img.size(0)), CV_8UC3, img.data_ptr<uchar>());

    std::istringstream iss(label_text);
    std::string line;
    int line_num = 0;
    int line_height = 20;
    int base_line = 0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 1;
    cv::Scalar color = cv::Scalar(0, 0, 255); 

    while (std::getline(iss, line)) {
        int y = 30 + line_num * line_height;
        cv::putText(img_mat, line, cv::Point(10, y), font_face,
                    font_scale, color, thickness);
        line_num++;
    }

    cv::imshow("Image with Prediction", img_mat);
    cv::waitKey(0);
}