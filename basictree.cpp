#include <cmath>
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
#include <utility>

#define SQUARE(x) ((x) * (x))

using namespace std;

struct Decision;
struct Leaf;

using Node = variant<Decision, Leaf>;

struct Decision {
    size_t feature;
    double threshold;
    unique_ptr<Node> left;
    unique_ptr<Node> right;
};

struct Leaf {
    double residual;
};

// Normalization metadata
struct NormMeta {
    vector<double> mean;
    vector<double> stdev;
};

namespace EnumEval {
    template<typename... Ts>
    struct overloaded: Ts... { using Ts::operator()...; };
}

class Dataset {
        void prep_normalization() {
            normalization = make_shared<NormMeta>(vector<double>(feature_count), vector<double>(feature_count));
            for (int current_feature = 0; current_feature < feature_count; current_feature++) {
                double mean = 0;

                for (int dp = 0; dp < dp_count; dp++) {
                    mean += features.at(current_feature).at(dp);
                }
                mean /= dp_count;
                double stdev = 0;
                for (int dp = 0; dp < dp_count; dp++) {
                    double difference = mean -features.at(current_feature).at(dp);
                    stdev += SQUARE(difference);
                }
                stdev /= dp_count;
                stdev = sqrt(stdev);
                normalization->mean[current_feature] = mean;
                normalization->stdev[current_feature] = stdev;
            }
        }

        // Because the test set needs to be normalized based on the training set, and they are split before normalization, the test set will inherit normalization metadata and only apply it, instead of calculating its own
        void apply_normalization() {
            for (int i = 0; i < feature_count; i++) {
                print("{:f}, ", normalization->mean.at(i));
            }
            println();
            for (int current_feature = 0; current_feature < feature_count; current_feature++) {
                for (int dp = 0; dp < dp_count; dp++) {
                    features[current_feature][dp] = (features[current_feature][dp] - normalization->mean.at(current_feature))/normalization->stdev.at(current_feature);
                }
            }

            normalized = true;
        }

    public:
        int dp_count;
        int feature_count;
        vector<vector<double>> features;
        vector<bool> labels;

        string filename;
        shared_ptr<NormMeta> normalization;
        bool normalized = false;
        bool labelled = false;
        unordered_map<string, int> feature_index_to_name;
        bool training_flag = true;

        // Dataset(bool training_flag = true): training_flag(training_flag) {}

        Dataset(const string& filename): filename(filename) {
            println("Creating dataset by file");
            ifstream in(filename);
            string line;

            int data_point = -1;
            int feature = 0;
            while (getline(in, line)) {

                auto tokens = line
                    | views::split(',')
                    | ranges::to<vector<string>>();

                    if (data_point == -1) {
                        features.resize(tokens.size()); 
                    }
                    for (auto& token : tokens) {
                    //println("{}", token);
                    // println("Adding  to features[{:d}]",  feature);
                    if (data_point == -1) {
                        feature_index_to_name[token] = feature;
                    } else {
                       features[feature].push_back(stod(token));
                    };
                    feature++;
                }

                // println("New line");



                feature = 0;
                data_point++;
            }
            feature_count = features.size();
            dp_count = data_point;
            println("Dataset created by file {:s}", filename);
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

        void label_by_feature(const string& label_feature) {
            if (!feature_index_to_name.contains(label_feature)) {
               println("The label doesn't exist in the data.");
               exit(-1);
            }
            int feature_index = feature_index_to_name.at(label_feature);
            // println("We will be using values from feature index {:d}", feature_index);
            // display_1_feature(feature_index);
            labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                labels[i] = (bool) features[feature_index][i];       
            }
            feature_index_to_name.erase(label_feature);
            features.erase(features.begin() + feature_index);
            feature_count--;
            labelled = true;
            println("Labelled dataset by feature {:s}", label_feature);
        }

        void label_rand() {
            labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                labels[i] = rand() % 2;
            }

            labelled = true;
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

            labelled = true;
        }

        void display() {
            for (int i = 0; i < feature_count; i++) {
                print("Feature {:d}: [", i);
                for (int j = 0; j < dp_count; j++) {
                    print("{:f}, ", features.at(i).at(j));
                }
                println("]");
            }

            print("\n[");
            for (int i = 0; i < dp_count; i++) {
                print("{:d}, ", labels.at(i));
            }
            println("]");
        }

        void display_1_feature(int whichone) {
            print("[");
            for (int i = 0; i < dp_count; i++) {
                print("{:f}, ", features.at(whichone).at(i));
            }
            println("]");
        }

        void show_feature_names() {
            for (const auto& pair : feature_index_to_name) {
                println("{:s}, ", pair.first);
            }
        }

        unique_ptr<vector<string>> get_features() {
            unique_ptr<vector<string>> feature_names = make_unique<vector<string>>();
            for (const auto& pair : feature_index_to_name) {
                println("{:s}", pair.first);
                feature_names->push_back(pair.first);
            }
            return feature_names;
        }

        const vector<bool>& get_labels() {
            return labels;
        }

        void display_1(int whichone = 0) {
            // println("Showing 1/{:d} datapoints", dp_count);
            print("[");
            for (int j = 0; j < feature_count; j++) {
               println("Features[{:d}] size = ", features.at(j).size());
                print("{:f}, ", features.at(j).at(whichone));
            }
            println("]");
        }

        void combine(Dataset& other) {
            unique_ptr<vector<string>> my_feature_names    = get_features();
            unique_ptr<vector<string>> other_feature_names = other.get_features();
            sort(my_feature_names->begin(), my_feature_names->end());
            sort(other_feature_names->begin(), other_feature_names->end());
            if (*my_feature_names != *other_feature_names) {
                println("Error joining datasets: features unequal");
                exit(-1);
            }

            if (!((labelled && other.labelled) || (!labelled && !other.labelled))) {
                println("Error joining datasets: labels unequal");
                exit(-1);
            }

            for (int i = 0; i < other.feature_count; i++) {
                this->features.at(i).insert(
                    this->features.at(i).end(),
                    other.features.at(i).begin(),
                    other.features.at(i).end()
                );
                println("{:d}", features.at(i).size());
            }
            const vector<bool>& other_labels = other.get_labels();
            labels.insert(labels.end(), other_labels.begin(), other_labels.end());
            dp_count += other.dp_count;
        }

        unique_ptr<Dataset> split(float train_percent) {
            if (train_percent < 0 || train_percent > 1) {
                println("Please enter a valid percentage (0.XX)");
                exit(-1);
            }
            int train_count = ceil((float)dp_count * train_percent);
            int test_count = dp_count - train_count;
            println("{:d} lines of training data", train_count);

            unique_ptr<Dataset> test  = make_unique<Dataset>(*this);  // copies everything

            // then overwrite just the parts that differ
            test->features.resize(feature_count);
            test->training_flag = false;

            for (int f = 0; f < feature_count; f++) {
                auto full = features[f];
                features[f]       = vector<double>(full.begin(), full.begin() + train_count);
                test->features[f] = vector<double>(full.begin() + train_count, full.end());
            }
            features.resize(feature_count);

            dp_count = train_count;
            test->dp_count  = test_count;

            // auto full_labels = labels;
            // labels = vector<bool>(labels.begin(), labels.begin() + train_count);
            // test->labels  = vector<bool>(full_labels.begin() + train_count, full_labels.end());

            return test;
        }

        void normalize() {
            if (training_flag) {
                prep_normalization();
                apply_normalization();
            } else {
                apply_normalization();
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

// double traverse(Node& head, Dataset& dataset) {
//     println("traversing");

//     return visit(EnumEval::overloaded{
//         [features=&dataset.features](Decision& decision) {
//             println("It was a decision node");
//             if (features->at(decision.feature) < decision.threshold) {
//                 println("We go down  the left line");
//                 return traverse(*decision.left, features);
//             } else {
//                 println("We go down  the right line");
//                 return traverse(*decision.right, features);
//             }
//         },
//         [](Leaf& leaf) {
//             println("Yup it ");

//             return leaf.residual;
//         },
//     }, head);
// }


IdealVariance findIdealVariance(Dataset& dataset) {
    double chosen_threshold = -1;
    int chosen_feature = -1;
    double min_variance = numeric_limits<double>::max();

    // For each feature

    #pragma omp parallel for
    for (int current_feature = 0; current_feature < dataset.feature_count; current_feature++) {
        int left_count;
        int right_count;
        // println("Looking at feature {:d}", current_feature);
        // Build a vector of indices of the datapoints sorted based on the current feature
        
        vector<pair<double, bool>> pairs(dataset.dp_count);
        for (int i = 0; i < dataset.dp_count; i++) {
            pairs[i] = {dataset.features.at(current_feature).at(i), dataset.labels.at(i)};
        }
        sort(pairs.begin(), pairs.end());

        // print("[");
        // for (pair pear : pairs) {
        //     print("{:f}, ",  pear.first);
        // }
        // println("]");

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

            // println("Splitting feature {:d} with {:d} on the left", current_feature, left_count);

            double value = pairs.at(feature_split_index).second;
            labels.left += value;
            labels.right -=  value;
            labels_squares.left += value*value;
            labels_squares.right -= value*value;
            // Does this actually do anything?
            if (left_count != dataset.dp_count && pairs[feature_split_index].first == pairs[feature_split_index+1].first) continue;

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
            //     println("End of feature {:d} (everything is in one category)", current_feature);
            // }

            double left_weight = (double) left_count / dataset.dp_count;
            double right_weight = (double) right_count / dataset.dp_count;

            double weighted_variance = (left_weight)*total_variance_left + (right_weight)*total_variance_right;
            // println("({:f})({:f}) + ({:f})({:f})", left_weight,  total_variance_left, right_weight, total_variance_right);
            #pragma omp critical
            {
                if (weighted_variance < min_variance) {
                    min_variance = weighted_variance;
                    chosen_threshold = pairs.at(feature_split_index).first;
                    chosen_feature = current_feature;
                }
            }
            

            
        }
    }
    println("The minimum variance was feature {:d} @ {:f}", chosen_feature, chosen_threshold);
    return IdealVariance{chosen_feature, chosen_threshold};
}
 
int main() {
    unique_ptr<Node> branch_0a_0a = make_unique<Node>(Leaf{60});
    unique_ptr<Node> branch_0a_0b = make_unique<Node>(Leaf{-60});


    unique_ptr<Node> branch_0a = make_unique<Node>(Decision{1, 0.5, move(branch_0a_0a), move(branch_0a_0b)});

    unique_ptr<Node> branch_0b = make_unique<Node>(Leaf{30});
    unique_ptr<Node> head = make_unique<Node>(Decision{0, 0.5, move(branch_0a), move(branch_0b)});

    vector<double> features = {0.1, 0.9, 0.9};
    println("Created structure, parsing");
    // double result = traverse(*head, features);
    // println("The result is {:g}", result);

    // unique_ptr<Dataset> dataset = make_unique<Dataset>(120000,15 );
    // dataset->label(5, 0.7);

    
    auto start = chrono::high_resolution_clock::now();
    unique_ptr<Dataset> dataset = make_unique<Dataset>("data/fragmented/1.csv"); 
    auto end = chrono::high_resolution_clock::now();
    println("Took {:d}ms", chrono::duration_cast<chrono::milliseconds>(end-start).count());

    dataset->display_1(43571); 
    dataset->show_feature_names();
    // dataset->label_by_feature("Class");


    unique_ptr<Dataset> test_dataset = make_unique<Dataset>("data/fragmented/2.csv");
    println("Test dataset has {:d} datapoints", test_dataset->dp_count);
    println("OG dataset has {:d} ({:d}) datapoints", dataset->dp_count, dataset->features.at(0).size());

    dataset->combine(*test_dataset);
    println("{:d}", dataset->dp_count);
    dataset->display_1(78376);

    unique_ptr<Dataset> test_set = dataset->split(0.8);
    dataset->label_by_feature("Class");
    test_set->label_by_feature("Class");

    dataset->normalize();
    println("Normalized dataset");
    test_set->normalization = dataset->normalization; // NOW share the pointer
    test_set->normalize();
    test_set->display_1(15);
    // int class_index = dataset->feature_index_to_name.at("Class");
    // println("Class index: {:d}", class_index);
    // for (int i = 0; i < dataset->dp_count; i++) {
    //     if (dataset->features.at(class_index).at(i) != dataset->labels.at(i)) {
    //         println("Wrong");
    //     }
    // }
}

    // IdealVariance thebest = findIdealVariance(*dataset);
