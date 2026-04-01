#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <utility>
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
#include <stdexcept>

#define SQUARE(x) ((x) * (x))
#define LEAF_SIZE 5
#define MAX_DEPTH 3
#define LAMBDA 1
#define LEARNING_RATE 0.01
#define NUM_TREES 1000
#define LOG_FILE "log.json"

#define SIGMOID(raw) (1.0 / (1.0 + exp(-raw)))

#define TREEGENSTATS_FIELDS     \
    SCALAR(int,    depth)       \
    SCALAR(int,    num_leaves)  \
    SCALAR(int,    num_decisions) \
    SCALAR(double, mean_leaf_value) \
    SCALAR(double, mean_variance) \
    SCALAR(double, MSE) \
    SCALAR(double, running_accuracy)

#define FEATUREDATA_FIELDS \
    VEC_SCALAR(double, feature_importance)

#define TRAININGLOG_FIELDS \
    STRUCT_ARRAY(TreeGenStats,  tree_stats)   \
    NESTED_STRUCT(FeatureData,  feature_data)

using namespace std;


void appendToFile(const std::string& filepath, const std::string& text) {
    std::ofstream file(filepath, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    file << text;
}

struct Decision;
struct Leaf;

using Node = variant<Decision, Leaf>;
class Dataset;

shared_ptr<Node> build_tree(Dataset&, int);
double traverse(Node&, const vector<double>&);

struct Decision {
    size_t feature;
    double threshold;
    shared_ptr<Node> left;
    shared_ptr<Node> right;
};

struct Leaf {
    double mean_label;
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

struct IdealVariance {
    int chosen_feature;
    double chosen_threshold;
    float left_count_vs_total;
    int left_count;
    double total_weighted_variance;
    pair<double, double> variances;
};

class BureauOfComplaints {
    static vector<string> important_registry;
    public:
        static int complain(const string& important_complaint) {
            important_registry.push_back(important_complaint);
            return 0;
        }
};
vector<string> BureauOfComplaints::important_registry;


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
        vector<double> true_labels;
        vector<double> residuals;
        vector<double> predictions;

        string filename;
        shared_ptr<NormMeta> normalization;
        bool normalized = false;
        bool labelled = false;
        unordered_map<string, int> feature_index_to_name;
        bool training_flag = true;
        double initial_prediction;

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
            feature_count = (int) features.size();
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

        void set_initial_prediction() {
            if (!labelled) {
                println("Only calculate initial predictions on labelled data!");
                exit(-1);
            }
            double true_label_mean  = accumulate(true_labels.begin(), true_labels.end(), 0.0)/dp_count;
            predictions.resize(dp_count);
            residuals.resize(dp_count);

            initial_prediction = log(true_label_mean/(1-true_label_mean));
            for (int dp_index = 0; dp_index < dp_count; dp_index++) {
                predictions.at(dp_index) = initial_prediction;
                residuals.at(dp_index) = true_labels.at(dp_index) - SIGMOID(initial_prediction);
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
            true_labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                true_labels[i] = (bool) features[feature_index][i];       
            }
            feature_index_to_name.erase(label_feature);
            features.erase(features.begin() + feature_index);
            feature_count--;
            labelled = true;
            println("Labelled dataset by feature {:s}", label_feature);
            set_initial_prediction();
            println("Set dataset initial predictions and residuals");
        }

        void label_rand() {
            true_labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                true_labels[i] = rand() % 2;
            }

            labelled = true;
            set_initial_prediction();
            println("Set dataset initial predictions and residuals");
        }

        void label(int feature, double threshold) {
            true_labels.resize(dp_count);
            for (int i = 0; i < dp_count; i++) {
                if (features.at(feature).at(i) < threshold) {
                    true_labels[i] = 0;
                } else {
                    true_labels[i] = 1;
                }
            }

            labelled = true;
            set_initial_prediction();
            println("Set dataset initial predictions and residuals");
        }

        void label_custom1(const string& open, const string& close) {
            if (!feature_index_to_name.contains(open) || !feature_index_to_name.contains(close)) {
                println("Check the label columns you provided to label_custom1 :(");
                exit(-1);
            }
            true_labels.resize(dp_count);
            for (int dp_index = 0; dp_index < dp_count-1; dp_index++) {
                true_labels.at(dp_index) = features.at(feature_index_to_name.at(close)).at(dp_index+1) - features.at(feature_index_to_name.at(open)).at(dp_index+1) > 0;
            }
            labelled = true;
            set_initial_prediction();
            println("Set dataset initial predictions and residuals");
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
                print("{:f}, ", true_labels.at(i));
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

        const vector<double>& get_labels() {
            return true_labels;
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
            const vector<double>& other_labels = other.get_labels();
            true_labels.insert(true_labels.end(), other_labels.begin(), other_labels.end());
            dp_count += other.dp_count;
        }

        shared_ptr<Dataset> split(float train_percent) {
            if (train_percent < 0 || train_percent > 1) {
                println("Please enter a valid percentage (0.XX)");
                exit(-1);
            }
            int train_count = ceil((float)dp_count * train_percent);
            int test_count = dp_count - train_count;
            println("{:d} lines of training data", train_count);

            shared_ptr<Dataset> test  = make_unique<Dataset>(*this);  // copies everything

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

        pair<shared_ptr<Dataset>, shared_ptr<Dataset>> split_by_criteria(shared_ptr<IdealVariance> criteria) {
            // println("THe left side is {}, the criteria was {} for feature {}", criteria->left_count, criteria->chosen_threshold, criteria->chosen_feature);
            shared_ptr<Dataset> left  = make_shared<Dataset>(*this);
            shared_ptr<Dataset> right = make_shared<Dataset>(*this);

            left->residuals.clear();
            right->residuals.clear();

            left->predictions.clear();
            right->predictions.clear();

            left->true_labels.clear();
            right->true_labels.clear();

            left->features.resize(feature_count);
            right->features.resize(feature_count);
            
            for (int f = 0; f < feature_count; f++) {
                left->features[f].clear();
                right->features[f].clear();
            }

            for (int dp_index = 0; dp_index < dp_count; dp_index++) {
                if (features.at(criteria->chosen_feature).at(dp_index) <= criteria->chosen_threshold) {
                    for (int f = 0; f < feature_count; f++) {
                        left->features[f].push_back(features[f][dp_index]);
                    }
                    if (labelled) left->true_labels.push_back(true_labels[dp_index]);
                    left->residuals.push_back(residuals[dp_index]);
                    left->predictions.push_back(predictions[dp_index]);

                } else {
                    for (int f = 0; f < feature_count; f++) {
                        right->features[f].push_back(features[f][dp_index]);
                    }
                    if (labelled) right->true_labels.push_back(true_labels[dp_index]);
                    right->residuals.push_back(residuals[dp_index]);
                    right->predictions.push_back(predictions[dp_index]);
                }
            }

            left->dp_count  = (int) left->features[0].size();
            right->dp_count = (int) right->features[0].size();
            // println("Split it to two datasets with sizes {}, {}", left->dp_count,right->dp_count);

            return {std::move(left), std::move(right)};
        }

        void normalize() {
            if (training_flag) {
                prep_normalization();
                apply_normalization();
            } else {
                apply_normalization();
            }
        }

        float class_split() {
            println("Warning: class split will not work on a regression model");
            if (!labelled) {
                println("You cannot calculate the class split on unlabelled data.");
                exit(-1);
            }

            pair<int, int> up_down;
            for (bool label : true_labels) {
                if (label) {
                    up_down.first++;
                }
                else {
                    up_down.second++;
                }
            }
            return (float)up_down.first / (float)up_down.second;
        }
};


struct Sum {
    double left = 0;
    double right = 0;
};

double traverse(Node& head, const vector<double>& features) {
    // println("traversing");

    return visit(EnumEval::overloaded{
        [features=features](Decision& decision) {
            // println("It was a decision node");
            if (features.at(decision.feature) < decision.threshold) {
                // println("We go down  the left line");
                return traverse(*decision.left, features);
            } else {
                //  println("We go down  the right line");
                return traverse(*decision.right, features);
            }
        },
        [](Leaf& leaf) {
           //  println("Yup it ");

            return leaf.mean_label;
        },
    }, head);
}


struct GradientBoosterMetadata {
    float learning_rate;
    int min_leaf_size;
    int max_depth;
    int num_trees;
};

struct TreeGenStats {
    #define SCALAR(type, name) type name;
    TREEGENSTATS_FIELDS
    #undef SCALAR

    string to_json(const string& pad = "  ") const {
        vector<string> entries;
        #define SCALAR(type, name) \
            entries.push_back(pad + "  \"" #name "\": " + to_string(name));
        TREEGENSTATS_FIELDS
        #undef SCALAR

        string s = pad + "{\n";
        for (size_t i = 0; i < entries.size(); i++) {
            s += entries[i];
            s += (i + 1 < entries.size()) ? ",\n" : "\n";
        }
        return s + pad + "}";
    }
};

struct FeatureData {
    #define VEC_SCALAR(type, name) vector<type> name;
    FEATUREDATA_FIELDS
    #undef VEC_SCALAR

    string to_json(const string& pad = "  ") const {
        vector<string> entries;
        #define VEC_SCALAR(type, name) {                        \
            string arr = pad + "  \"" #name "\": [";           \
            for (size_t i = 0; i < (name).size(); i++) {         \
                arr += to_string((name)[i]);                     \
                if (i + 1 < (name).size()) arr += ", ";          \
            }                                                   \
            entries.push_back(arr + "]");                      \
        }
        FEATUREDATA_FIELDS
        #undef VEC_SCALAR

        string s = pad + "{\n";
        for (size_t i = 0; i < entries.size(); i++) {
            s += entries[i];
            s += (i + 1 < entries.size()) ? ",\n" : "\n";
        }
        return s + pad + "}";
    }
};

struct TrainingLog {
    #define STRUCT_ARRAY(type, name)  vector<type> name;
    #define NESTED_STRUCT(type, name) type name;
    TRAININGLOG_FIELDS
    #undef STRUCT_ARRAY
    #undef NESTED_STRUCT

    string to_json() const {
        vector<string> entries;

        #define STRUCT_ARRAY(type, name) {                          \
            string arr = "  \"" #name "\": [\n";                   \
            for (size_t i = 0; i < (name).size(); i++) {             \
                arr += (name)[i].to_json("    ");                    \
                arr += (i + 1 < (name).size()) ? ",\n" : "\n";      \
            }                                                       \
            entries.push_back(arr + "  ]");                        \
        }
        #define NESTED_STRUCT(type, name) \
            entries.push_back("  \"" #name "\": " + (name).to_json("  "));
        TRAININGLOG_FIELDS
        #undef STRUCT_ARRAY
        #undef NESTED_STRUCT

        string s = "{\n";
        for (size_t i = 0; i < entries.size(); i++) {
            s += entries[i];
            s += (i + 1 < entries.size()) ? ",\n" : "\n";
        }
        return s + "}\n";
    }
};

void write_training_log(const shared_ptr<TrainingLog>& log) {
    ofstream out(LOG_FILE, ios::out | ios::trunc);
    if (!out.is_open()) {
        println("Failed to open log file: {}", LOG_FILE);
        return;
    }
    out << log->to_json();
    out.close();
    println("Training log written to {}", LOG_FILE);
}


class GradientBooster {
    vector<shared_ptr<Node>> trees;
    shared_ptr<Dataset> train_dataset;
    shared_ptr<Dataset> test_dataset;
    
    public:
        TrainingLog training_log;
        float learning_rate;
        int min_leaf_size;
        int max_depth;
        int num_trees;
    
        GradientBooster(const GradientBoosterMetadata& gbm, shared_ptr<Dataset> train_dataset, shared_ptr<Dataset> test_dataset): 
            train_dataset(std::move(train_dataset)),
            test_dataset(std::move(test_dataset)), 
            learning_rate(gbm.learning_rate), 
            min_leaf_size(gbm.min_leaf_size), 
            max_depth(gbm.max_depth),
            num_trees(gbm.num_trees)
            {
                println("Gradient Booster Initializing");
        }

        shared_ptr<Node> build_tree(Dataset& dataset, int tree_index, double variance, int current_depth=1) {
            // Leaf base case
            if (dataset.dp_count <= 2*min_leaf_size  || current_depth == MAX_DEPTH) { // ||  variance < 0.02

                // Calculate the mean residual
                double grad_sum = 0, hess_sum = 0;
                for (int i = 0; i < dataset.dp_count; i++) {
                    double p_i = SIGMOID(dataset.predictions.at(i));
                    grad_sum += dataset.true_labels.at(i) - p_i;
                    hess_sum += p_i * (1.0 - p_i);
                }

                double leaf_value = grad_sum / (hess_sum + LAMBDA);


                training_log.tree_stats.at(tree_index).mean_leaf_value += leaf_value;
                // println("Leaf value: {:f}, grad sum {:f}, dp count {:d}", leaf_value, grad_sum, dataset.dp_count);
                // Increment leaf count for this tree
                training_log.tree_stats.at(tree_index).num_leaves++;
                // Make sure we record the deepest depth in the tree
                if (current_depth > training_log.tree_stats.at(tree_index).depth) {
                    training_log.tree_stats.at(tree_index).depth = current_depth;
                }

                return make_shared<Node>(
                    Leaf{
                        leaf_value
                    }
                );
            }

            // Find the ideal split and split the dataset
            // println("FInding variance::=======\n");
            shared_ptr<IdealVariance> ideal_variance = findIdealVariance(dataset);
            // println("The variance has been found, column was {}", ideal_variance->chosen_feature);
            training_log.feature_data.feature_importance.at(ideal_variance->chosen_feature)++;
            training_log.tree_stats.at(tree_index).mean_variance += ideal_variance->total_weighted_variance;
            // println("Ideal variance left size: {:f}", ideal_variance->left_count_vs_total);
            auto [left_dataset, right_dataset] = dataset.split_by_criteria(ideal_variance);
            // println("\n========Split by criteria found in variance");
            //.println("The size of the left and right datasets is {:d} and {:d}", left_dataset->dp_count, right_dataset->dp_count);
            // Build left and right trees based on their datasets
            shared_ptr<Node> new_left = build_tree(*left_dataset, tree_index, ideal_variance->variances.first, current_depth+1);
            shared_ptr<Node> new_right = build_tree(*right_dataset, tree_index, ideal_variance->variances.second, current_depth+1);

            training_log.tree_stats.at(tree_index).num_decisions++;
            shared_ptr<Node> new_decision = make_shared<Node>(
                Decision { 
                    (size_t)ideal_variance->chosen_feature,
                    ideal_variance->chosen_threshold,
                    new_left,
                    new_right
                }
            );


            return new_decision;    
        }

        void generate(int current_tree_count = 1) {
            // Passing a big number as the initial variance so it definitely doesn't trigger the low-variance leaf creation
            shared_ptr<Node> current_tree = build_tree(*train_dataset, current_tree_count-1, numeric_limits<double>::max(), 1); 

            auto& stats = training_log.tree_stats.at(current_tree_count - 1);
            if (stats.num_decisions > 0)
                stats.mean_variance   /= stats.num_decisions;
            if (stats.num_leaves > 0)
                stats.mean_leaf_value /= stats.num_leaves;

            // println("Built one tree\n");
            trees.push_back(current_tree);
            int feature_count = train_dataset->feature_count;
            int dp_count = train_dataset->dp_count;
            double MSE = 0;

            //println("About to run data points through the tree at index {:d}", current_tree_count);
            float running_accuracy = 0;
            for (int dp_index = 0; dp_index < dp_count; dp_index++) {
               //   println("\n\nData point {:d} being evaluated", dp_index+1);
                vector<double> features_temp;
                for (int feature_index = 0; feature_index < feature_count; feature_index++) {
                    //println("About to access the data set at feature  {:d} of dp {:d}", feature_)
                    features_temp.push_back(train_dataset->features.at(feature_index).at(dp_index));
                }
                // println("features for that one data point collected collected");
                double leaf_value = traverse(*current_tree, features_temp);
                // println("===============\nTRAVERSED and got {:f}\n======================", leaf_value);
                double prediction_update = learning_rate*leaf_value;
                // println("Log odd prediction updated from {:f} to {:f}", train_dataset->predictions.at(dp_index), train_dataset->predictions.at(dp_index)+prediction_update);
                train_dataset->predictions.at(dp_index) += prediction_update;
                double p_i = SIGMOID(train_dataset->predictions.at(dp_index));  // sigmoid
                double residual = train_dataset->true_labels.at(dp_index) - p_i;
                train_dataset->residuals.at(dp_index) = residual;
                MSE += SQUARE(residual);
                // !non-regression
                if ((train_dataset->predictions.at(dp_index) >= 0.5 && (bool)(train_dataset->true_labels.at(dp_index))) || (train_dataset->predictions.at(dp_index) < 0.5 && !(bool)(train_dataset->true_labels.at(dp_index)))) {
                    running_accuracy += 1;
                }
                //println("Updated the residual to {}", train_dataset->residuals.at(dp_index));
                //appendToFile("log.txt", format());
                // gradient
                //train_dataset->residuals.at(dp_index) = train_dataset->true_labels.at(dp_index) - new_prediction;
            }
            running_accuracy /= (float)dp_count;
            training_log.tree_stats.at(current_tree_count-1).running_accuracy = running_accuracy;
            MSE /= dp_count;
            stats.MSE = MSE;


            println("Evaluated data points at tree {:d} ({}) ({})\n\n", current_tree_count, MSE, training_log.tree_stats.at(current_tree_count-1).mean_variance);

            if (current_tree_count == num_trees) {
                return;
            }

            generate(current_tree_count+1);
        }

        shared_ptr<TrainingLog> train() {
            training_log.feature_data.feature_importance.resize(train_dataset->feature_count);
            training_log.tree_stats.resize(num_trees);
            // println("Generating");
            generate();
            return make_shared<TrainingLog>(training_log);
        }

        double infer(const vector<double>& features, double initial) {
            double result = initial;
            for (int tree_index = 0; tree_index < num_trees; tree_index++) {
                result += traverse(*(trees.at(tree_index)), features);
            }
            return result;
        }

        double test() {
            int correct_predictions = 0;
            for (int dp_index = 0; dp_index < test_dataset->dp_count; dp_index++) {
                vector<double> features(test_dataset->feature_count);
                for (int j = 0; j < test_dataset->feature_count; j++) {
                    features.at(j) = test_dataset->features.at(j).at(dp_index);
                }
                double inference = infer(features, test_dataset->initial_prediction);
                double probability = SIGMOID(inference);
                // !non-regression
                if ((probability >= 0.5 && (bool)(test_dataset->true_labels.at(dp_index))) || (probability < 0.5 && !(bool)(test_dataset->true_labels.at(dp_index)))) {
                    correct_predictions++;
                }

            }
            return (double)correct_predictions/test_dataset->dp_count;
        }

        shared_ptr<IdealVariance> findIdealVariance(Dataset& dataset) {
            int subset_size = max(1, (int)sqrt(dataset.feature_count));
            
            vector<int> feature_indices(dataset.feature_count);
            iota(feature_indices.begin(), feature_indices.end(), 0);
            shuffle(feature_indices.begin(), feature_indices.end(), default_random_engine{random_device{}()});
            feature_indices.resize(subset_size);

            double chosen_threshold = -1;
            int chosen_feature = -1;
            double min_variance = numeric_limits<double>::max();
            int dataset_size = dataset.dp_count;
            int optimal_left_count = 0;

            double left_variance = 0;
            double right_variance = 0;

            if (dataset_size < 2 * min_leaf_size) {
                println("Its a little small. This probably should have been caught earlier though");
                exit(-1);
            }

            #pragma omp parallel for
            for (int current_feature_index = 0; current_feature_index < subset_size; current_feature_index++) {
                int current_feature = feature_indices.at(current_feature_index);
                
                vector<pair<double, double>> pairs(dataset.dp_count);
                for (int i = 0; i < dataset.dp_count; i++) {
                    pairs[i] = {dataset.features.at(current_feature).at(i), dataset.residuals.at(i)};
                }
                sort(pairs.begin(), pairs.end());

                Sum labels;
                Sum labels_squares;
                for (int temp = 0; temp < dataset.dp_count; temp++) {
                    double value = pairs.at(temp).second;
                    labels.right += value;
                    labels_squares.right += value * value;
                }

                //  Pre-fill the first (min_leaf_size - 1) elements
                for (int i = 0; i < min_leaf_size - 1; i++) {
                    double value = pairs.at(i).second;
                    labels.left += value;
                    labels.right -= value;
                    labels_squares.left += value * value;
                    labels_squares.right -= value * value;
                }

                // Only iterate through indices that satisfy both min_leaf_size constraints
                // Start index: min_leaf_size - 1 (the index that results in min_leaf_size on left)
                // End index: dataset_size - min_leaf_size - 1 (results in min_leaf_size on right)
                for (int feature_split_index = min_leaf_size - 1; feature_split_index <= dataset_size - min_leaf_size; feature_split_index++) {
                    int left_count = feature_split_index + 1;
                    int right_count = dataset_size - left_count;

                    double value = pairs.at(feature_split_index).second;
                    labels.left += value;
                    labels.right -= value;
                    labels_squares.left += value * value;
                    labels_squares.right -= value * value;

                    // Skip identical feature values to prevent splitting between same numbers
                    if (feature_split_index < dataset_size - 1 && pairs[feature_split_index].first == pairs[feature_split_index + 1].first) {
                        continue;
                    }

                    double total_variance_left = (labels_squares.left / left_count) - SQUARE(labels.left / left_count);
                    double total_variance_right = (labels_squares.right / right_count) - SQUARE(labels.right / right_count);

                    double weighted_variance = ((double)left_count / dataset_size) * total_variance_left + 
                                            ((double)right_count / dataset_size) * total_variance_right;

                    #pragma omp critical
                    {
                        if (weighted_variance < min_variance) {
                            optimal_left_count = left_count;
                            min_variance = weighted_variance;
                            chosen_threshold = pairs.at(feature_split_index).first;
                            chosen_feature = current_feature;
                            left_variance = total_variance_left;
                            right_variance = total_variance_right;
                        }
                    }
                }
            }

            if (chosen_feature == -1) {
                println("The dataset passed into findIdealVariance was too small or no valid splits found.");
                exit(-1);
            }

            return make_shared<IdealVariance>(IdealVariance{chosen_feature, chosen_threshold, (float)optimal_left_count / (float)dataset_size, (int)optimal_left_count, min_variance, pair<double, double>(left_variance, right_variance)});
        }
};

// day 4, hour 15 + ,: amaz
//  day 5?  perhaps 22:37



template<typename F>
auto loki(F lambda) {
    auto start = chrono::high_resolution_clock::now();
    auto result = lambda();  // just use auto here
    auto end = chrono::high_resolution_clock::now();
    return make_pair(std::move(result), chrono::duration_cast<chrono::milliseconds>(end-start).count());
}

int manual_tree_test() {
    unique_ptr<Node> branch_0a_0a = make_unique<Node>(Leaf{60});
    unique_ptr<Node> branch_0a_0b = make_unique<Node>(Leaf{-60});


    unique_ptr<Node> branch_0a = make_unique<Node>(Decision{1, 0.5, std::move(branch_0a_0a), std::move(branch_0a_0b)});

    unique_ptr<Node> branch_0b = make_unique<Node>(Leaf{30});
    unique_ptr<Node> head = make_unique<Node>(Decision{0, 0.5, std::move(branch_0a), std::move(branch_0b)});

    vector<double> features = {0.1, 0.9, 0.9};
    println("Created structure, parsing");
    return 1;
}

int maine() {

    // double result = traverse(*head, features);
    // println("The result is {:g}", result);

  //  unique_ptr<Dataset> dataset = make_unique<Dataset>("data/fragmented/1.csv"); 

    auto [dataset, time] = loki(
        [](){
        return make_shared<Dataset>("data/fragmented/1.csv"); 
        }
    );

    println("Loading one dataset [:d ms]", time);
    
    //dataset->display_1(43571); 

    dataset->show_feature_names();
    // dataset->label_by_feature("Class");


    shared_ptr<Dataset> test_dataset = make_shared<Dataset>("data/fragmented/2.csv");
    println("Loaded test dataset");
    println("Test dataset has {:d} datapoints", test_dataset->dp_count);
    println("OG dataset has {:d} ({:d}) datapoints", dataset->dp_count, dataset->features.at(0).size());


    dataset->combine(*test_dataset);
    println("Combined datasets");
    println("{:d}", dataset->dp_count);
    //dataset->display_1(78376);

    float train_percent = 0.8;
    shared_ptr<Dataset> test_set = dataset->split(train_percent);
    println("Split dataset to train and test at {:f}%", train_percent*100);
    dataset->label_by_feature("Class");
    println("Labelled train dataset by class");
    test_set->label_by_feature("Class");
    println("Labelled test dataset by class");

    dataset->normalize();
    println("Normalized train dataset");
    test_set->normalization = dataset->normalization; // NOW share the pointer
    test_set->normalize();
    println("Normalized test dataset");
    test_set->display_1(15);

    GradientBoosterMetadata md{LEARNING_RATE, LEAF_SIZE, MAX_DEPTH, NUM_TREES};
    GradientBooster gb(md, dataset, test_set);
    shared_ptr<TrainingLog> log = gb.train();
    double accuracy = gb.test();
    println("Accuracy: {:f}", accuracy);

    ofstream outFile(LOG_FILE);
    outFile << log->to_json();
    outFile.close();



    // int class_index = dataset->feature_index_to_name.at("Class");
    // println("Class index: {:d}", class_index);
    // for (int i = 0; i < dataset->dp_count; i++) {
    //     if (dataset->features.at(class_index).at(i) != dataset->labels.at(i)) {
    //         println("Wrong");
    //     }
    // }
    return 0;
}

int mainee() {
    auto [dataset, time] = loki(
        [](){
        return make_shared<Dataset>("data/sp500/data.csv"); 
        }
    );
    println("Loading one dataset {:d} ms", time);
    dataset->label_custom1("open", "close");

    float train_percent = 0.8;
    shared_ptr<Dataset> test_set = dataset->split(train_percent);

    dataset->normalize();
    test_set->normalization = dataset->normalization; 
    test_set->normalize();

    GradientBoosterMetadata md{0.01, 10, 5, 600};
    GradientBooster gb(md, dataset, test_set);
    shared_ptr<TrainingLog> log = gb.train();
    double accuracy = gb.test();
    println("Accuracy: {:f}", accuracy);

    ofstream outFile(LOG_FILE);
    outFile << log->to_json();
    outFile.close();
    return 0;
}
//int 
    // IdealVariance thebest = findIdealVariance(*dataset);

int main() {
    auto [dataset, time] = loki(
        [](){
        return make_shared<Dataset>("data/generic/CLEAN_AAPL_PREPARED.csv"); 
        }
    );
    println("Loading one dataset {:d} ms", time);
    dataset->label_custom1("Open", "Close(t)");
    println("Class split: {}", dataset->class_split());

    float train_percent = 0.99;
    shared_ptr<Dataset> test_set = dataset->split(train_percent);

    dataset->normalize();
    test_set->normalization = dataset->normalization; 
    test_set->normalize();

    GradientBoosterMetadata md{LEARNING_RATE,LEAF_SIZE, MAX_DEPTH, NUM_TREES};
    GradientBooster gb(md, dataset, test_set);
    shared_ptr<TrainingLog> log = gb.train();
    double accuracy = gb.test();
    println("Accuracy: {:f}", accuracy);

    ofstream outFile(LOG_FILE);
    outFile << log->to_json();
    outFile.close();
    return 0;
}