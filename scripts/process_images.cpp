#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void processAndSaveImages(const std::string& input_path, const std::string& output_path) {
    for (const auto& planet_dir : fs::directory_iterator(input_path)) {
        if (planet_dir.is_directory()) {
            std::string planet_name = planet_dir.path().filename().string();
            std::string output_planet_dir = output_path + "/" + planet_name;

            fs::create_directories(output_planet_dir);

            for (const auto& img_entry : fs::directory_iterator(planet_dir.path())) {
                if (img_entry.is_regular_file()) {
                    std::string img_path = img_entry.path().string();
                    std::string output_img_path = output_planet_dir + "/" + img_entry.path().filename().string();

                    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
                    if (img.empty()) {
                        std::cerr << "Failed to read image: " << img_path << std::endl;
                        continue;
                    }

                    if (img.rows < 144 || img.cols < 144) {
                        std::cerr << "Image too small for cropping: " << img_path << std::endl;
                        continue;
                    }

                    cv::Rect crop_region(56, 0, 144, 144); // x=0, y=56, width=144, height=144
                    cv::Mat cropped_img = img(crop_region);

                    cv::Mat resized_img;
                    cv::resize(cropped_img, resized_img, cv::Size(256, 256));

                    if (!cv::imwrite(output_img_path, resized_img)) {
                        std::cerr << "Failed to save image: " << output_img_path << std::endl;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <output_folder>" << std::endl;
        return 1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    try {
        processAndSaveImages(input_folder, output_folder);
    } catch (const std::exception& e) {
        std::cerr << "Error processing images: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Processing completed successfully." << std::endl;
    return 0;
}
