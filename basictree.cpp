#include <string>
#include <variant>
#include <memory>
#include <vector>
#include <print>
#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>
#include <ranges>
#include <unordered_map>

#define SQUARE(x) ((x) * (x))

struct Decision;
struct Leaf;

using Node = std::variant<Decision, Leaf>;

struct Decision {
    size_t feature;
    double threshold;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

struct Leaf {
    double residual;
};

namespace EnumEval {
    template<typename... Ts>
    struct overloaded: Ts... { using Ts::operator()...; };
}

class Dataset {
    public:
        int dp_count;
        int feature_count;
        std::vector<std::vector<double>> features;
        std::vector<bool> labels;

        std::string filename;
        bool normalized = false;
        std::unordered_map<int, std::string> feature_index_to_name;

        Dataset(const std::string& filename): filename(filename) {
            std::println("Creating dataset by file");
            std::ifstream in(filename);
            std::string line;

            int data_point = -1;
            int feature = 0;
            while (std::getline(in, line)) {

                auto tokens = line
                    | std::views::split(',')
                    | std::ranges::to<std::vector<std::string>>();

                    if (data_point == -1) {
                        features.resize(tokens.size()); 
                    }
                    for (auto& token : tokens) {
                    //std::println("{}", token);
                    // std::println("Adding  to features[{:d}]",  feature);
                    if (data_point == -1) {
                        feature_index_to_name[feature] = token;
                    } else {
                       features[feature].push_back(std::stod(token));
                    };
                    feature++;
                }

                // std::println("New line");



                feature = 0;
                data_point++;
            }
            feature_count = features.size();
            dp_count = data_point;

        }

        Dataset(int dp_count, int feature_count): dp_count(dp_count), feature_count(feature_count) {
            filename = "dummy";
            // Each row is a list of features
            
            features.resize(feature_count);
            for (int i = 0; i < feature_count; i++) {
                features[i].resize(dp_count);
                for (int j = 0; j < dp_count; j++) {
                    features[i][j] = (double) rand() / RAND_MAX;
                }
            }

        }

        void label_rand() {
            labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                labels[i] = rand() % 2;
            }
        }

        void label(int feature, double threshold) {
            labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                if (features.at(feature).at(i) < threshold) {
                    labels[i] = 0;
                } else {
                    labels[i] = 1;
                }
            }
        }

        void display() {
            for (int i = 0; i < feature_count; i++) {
                std::print("Feature {:d}: [", i);
                for (int j = 0; j < dp_count; j++) {
                    std::print("{:f}, ", features.at(i).at(j));
                }
                std::println("]");
            }

            std::print("\n[");
            for (int i = 0; i < dp_count; i++) {
                std::print("{:d}, ", labels.at(i));
            }
            std::println("]");
        }

        void display_1(int whichone = 0) {
            std::println("Showing 1/{:d} datapoints", dp_count);
            std::print("[");
            for (int j = 0; j < feature_count; j++) {
                std::print("{:f}, ", features.at(j).at(whichone));
            }
            std::println("]");
            
        }

        void show_feature_names() {
            for (int i = 0; i < feature_count; i++) {
                std::println("{:s}, ", feature_index_to_name.at(i));
            }
        }

};

struct IdealVariance {
    int chosen_feature;
    double chosen_threshold;
};

struct Sum {
    double left = 0;
    double right = 0;
};

double traverse(Node& head, std::vector<double>& features) {
    std::println("traversing");

    return std::visit(EnumEval::overloaded{
        [&features](Decision& decision) {
            std::println("It was a decision node");
            if (features.at(decision.feature) < decision.threshold) {
                std::println("We go down  the left line");
                return traverse(*decision.left, features);
            } else {
                std::println("We go down  the right line");
                return traverse(*decision.right, features);
            }
        },
        [](Leaf& leaf) {
            std::println("Yup it ");

            return leaf.residual;
        },
    }, head);
}


IdealVariance findIdealVariance(Dataset& dataset) {
    double chosen_threshold = -1;
    int chosen_feature = -1;
    double min_variance = std::numeric_limits<double>::max();

    // For each feature
    int left_count;
    int right_count;
    #pragma omp parallel for
    for (int current_feature = 0; current_feature < dataset.feature_count; current_feature++) {
        // std::println("Looking at feature {:d}", current_feature);
        // Build a vector of indices of the datapoints sorted based on the current feature
        
        std::vector<std::pair<double, bool>> pairs(dataset.dp_count);
        for (int i = 0; i < dataset.dp_count; i++) {
            pairs[i] = {dataset.features.at(current_feature).at(i), dataset.labels.at(i)};
        }
        std::sort(pairs.begin(), pairs.end());

        // std::print("[");
        // for (std::pair pear : pairs) {
        //     std::print("{:f}, ",  pear.first);
        // }
        // std::println("]");

        // Label sums to calculate residuals
        Sum labels;
        Sum labels_squares;
        // Sum it all pre-emptively and store in the right sum to allow for easy manipulation later
        for (int temp = 0; temp < dataset.dp_count;  temp++) {
            double value = pairs.at(temp).second;
            labels.right += value;
            labels_squares.right += value*value;
        }

        for (int feature_split_index = 0; feature_split_index < dataset.dp_count; feature_split_index++) {
            left_count = feature_split_index + 1;
            right_count = dataset.dp_count - left_count;
            if (left_count != dataset.dp_count && pairs[feature_split_index].first == pairs[feature_split_index+1].first) continue;

            // std::println("Splitting feature {:d} with {:d} on the left", current_feature, left_count);

            double value = pairs.at(feature_split_index).second;
            labels.left += value;
            labels.right -=  value;
            labels_squares.left += value*value;
            labels_squares.right -= value*value;

            double mean_of_squares_left = labels_squares.left / (left_count);
            double square_of_means_left = SQUARE(labels.left / (left_count));
            double total_variance_left = mean_of_squares_left - square_of_means_left;

            double total_variance_right = 0;

            // If the index is at the last index, all of the values are in the left bucket
            if (right_count != 0) {
                double mean_of_squares_right = labels_squares.right / (right_count);
                double square_of_means_right = SQUARE(labels.right / (right_count));
                total_variance_right = mean_of_squares_right - square_of_means_right;
            } 
            // else {
            //     std::println("End of feature {:d} (everything is in one category)", current_feature);
            // }

            double left_weight = (double) left_count / dataset.dp_count;
            double right_weight = (double) right_count / dataset.dp_count;

            double weighted_variance = (left_weight)*total_variance_left + (right_weight)*total_variance_right;
            // std::println("({:f})({:f}) + ({:f})({:f})", left_weight,  total_variance_left, right_weight, total_variance_right);
            if (weighted_variance < min_variance) {
                min_variance = weighted_variance;
                chosen_threshold = pairs.at(feature_split_index).first;
                chosen_feature = current_feature;
            }
            

            
        }
    }
    std::println("The minimum variance was feature {:d} @ {:f}", chosen_feature, chosen_threshold);
    return IdealVariance{chosen_feature, chosen_threshold};
}

int main() {
    std::unique_ptr<Node> branch_0a_0a = std::make_unique<Node>(Leaf{60});
    std::unique_ptr<Node> branch_0a_0b = std::make_unique<Node>(Leaf{-60});


    std::unique_ptr<Node> branch_0a = std::make_unique<Node>(Decision{1, 0.5, std::move(branch_0a_0a), std::move(branch_0a_0b)});

    std::unique_ptr<Node> branch_0b = std::make_unique<Node>(Leaf{30});
    std::unique_ptr<Node> head = std::make_unique<Node>(Decision{0, 0.5, std::move(branch_0a), std::move(branch_0b)});

    std::vector<double> features = {0.1, 0.9, 0.9};
    std::println("Created structure, parsing");
    double result = traverse(*head, features);
    std::println("The result is {:g}", result);

    // std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>(120000,15 );
    // dataset->label(5, 0.7);

    
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("data/train.csv"); 
    auto end = std::chrono::high_resolution_clock::now();
    std::println("Took {:d}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());

    dataset->display_1(117563); 
    dataset->show_feature_names();
    

}

    // IdealVariance thebest = findIdealVariance(*dataset);
