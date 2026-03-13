#include <cmath>
#include <numeric>
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

#define SQUARE(x) ((x) * (x))
#define LEAF_SIZE 5
#define MAX_DEPTH 3

using namespace std;

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
            double true_label_mean  = accumulate(true_labels.begin(), true_labels.end(), 0)/dp_count;
            predictions.resize(dp_count);
            residuals.resize(dp_count);
            for (int dp_index = 0; dp_index < dp_count; dp_index++) {
                predictions.at(dp_index) = true_label_mean;
                residuals.at(dp_index) = true_labels.at(dp_index) - true_label_mean;
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

        pair<shared_ptr<Dataset>, shared_ptr<Dataset>> split_by_criteria(const IdealVariance& criteria) {
            shared_ptr<Dataset> left  = make_shared<Dataset>(*this);
            shared_ptr<Dataset> right = make_shared<Dataset>(*this);

            left->features.resize(feature_count);
            right->features.resize(feature_count);
            for (int f = 0; f < feature_count; f++) {
                left->features[f].clear();
                right->features[f].clear();
            }

            left->true_labels.clear();
            right->true_labels.clear();

            for (int i = 0; i < dp_count; i++) {
                if (features[criteria.chosen_feature][i] < criteria.chosen_threshold) {
                    for (int f = 0; f < feature_count; f++) {
                        left->features[f].push_back(features[f][i]);
                    }
                    if (labelled) left->true_labels.push_back(true_labels[i]);
                } else {
                    for (int f = 0; f < feature_count; f++) {
                        right->features[f].push_back(features[f][i]);
                    }
                    if (labelled) right->true_labels.push_back(true_labels[i]);
                }
            }

            left->dp_count  = (int) left->features[0].size();
            right->dp_count = (int) right->features[0].size();

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
};


struct Sum {
    double left = 0;
    double right = 0;
};

struct GradientBoosterMetadata {
    float learning_rate;
    int max_leaf_size;
    int max_depth;
    int num_trees;
};

class GradientBooster {
    vector<shared_ptr<Node>> trees;
    shared_ptr<Dataset> dataset;
    
    public:
        float learning_rate;
        int max_leaf_size;
        int max_depth;
        int num_trees;
    
        GradientBooster(const GradientBoosterMetadata& gbm, shared_ptr<Dataset> dataset): 
            dataset(std::move(dataset)), 
            learning_rate(gbm.learning_rate), 
            max_leaf_size(gbm.max_leaf_size), 
            max_depth(gbm.max_depth),
            num_trees(gbm.num_trees)
            {
                println("Gradient Booster Initializing");
        }

        void generate(int current_tree_count = 1) {
            shared_ptr<Node> current_tree = build_tree(*dataset, 1);
            println("Built one tree");
            trees.push_back(current_tree);
            int feature_count = dataset->feature_count;
            int dp_count = dataset->dp_count;

            println("About to run data points through it");

            for (int dp_index = 0; dp_index < dp_count; dp_index++) {
               //   println("\n\nData point {:d} being evaluated", dp_index+1);
                vector<double> features_temp(feature_count);
                for (int feature_index = 0; feature_index < feature_count; feature_index++) {
                    //println("About to access the data set at feature  {:d} of dp {:d}", feature_)
                    features_temp.push_back(dataset->features.at(feature_index).at(dp_index));
                }
                // println("features for that one data point collected collected");
                double leaf_value = traverse(*current_tree, features_temp);
                //  println("===============\nTRAVERSED\n======================");
                double new_prediction =  learning_rate*leaf_value;
                dataset->predictions.at(dp_index) += new_prediction;
                dataset->residuals.at(dp_index) = dataset->true_labels.at(dp_index) - new_prediction;
            }
            println("Evaluated data points at tree {:d}", current_tree_count);

            if (current_tree_count == num_trees) {
                return;
            }

            generate(current_tree_count+1);
        }
};

// day 4, hour 15 + ,: amaz

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


IdealVariance findIdealVariance(Dataset& dataset) {
    double chosen_threshold = -1;
    int chosen_feature = -1;
    double min_variance = numeric_limits<double>::max();
    int dataset_size = dataset.dp_count;
    int optimal_left_count = 0; // To keep track of left-count-vs-total later


    // For each feature

    #pragma omp parallel for
    for (int current_feature = 0; current_feature < dataset.feature_count; current_feature++) {
        int left_count;
        int right_count;
        // println("Looking at feature {:d}", current_feature);
        // Build a vector of indices of the datapoints sorted based on the current feature
        
        vector<pair<double, bool>> pairs(dataset.dp_count);
        for (int i = 0; i < dataset.dp_count; i++) {
            pairs[i] = {dataset.features.at(current_feature).at(i), dataset.residuals.at(i)};
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
                    optimal_left_count = left_count;
                    min_variance = weighted_variance;
                    chosen_threshold = pairs.at(feature_split_index).first;
                    chosen_feature = current_feature;
                }
            }
            

            
        }
    }
    // println("The minimum variance was feature {:d} @ {:f}", chosen_feature, chosen_threshold);
    return IdealVariance{chosen_feature, chosen_threshold, (float)optimal_left_count/(float)dataset_size};
}

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

int variance_n_p_hp(int dp_count, int feature_count) {
    unique_ptr<Dataset> dataset = make_unique<Dataset>(dp_count,feature_count);
    dataset->label(5, 0.7);
    auto [ideal_variance, time] = loki(
        [&dataset]() {
                 return findIdealVariance(*dataset);   
        }
    );
    println("One variance on tree {:d} data points and {:d} features takes {:d}ms", dp_count, feature_count, time);
    return 1;
}

shared_ptr<Node> build_tree(Dataset& dataset, int current_depth=1) {
    // Leaf base case
    if (dataset.dp_count < LEAF_SIZE || current_depth == MAX_DEPTH) {
        // Calculate the mean residual
        double current_mean_label = 0;
        for (double residual : dataset.residuals) {
            current_mean_label += residual;
        }
        current_mean_label /= dataset.dp_count;
        return make_shared<Node>(
            Leaf{ 
                current_mean_label
            }
        );
    }
    // Find the ideal split and split the dataset
    IdealVariance ideal_variance = findIdealVariance(dataset);
    auto [left_dataset, right_dataset] = dataset.split_by_criteria(ideal_variance);

    // Build left and right trees based on their datasets
    shared_ptr<Node> new_left = build_tree(*left_dataset, current_depth+1);
    shared_ptr<Node> new_right = build_tree(*right_dataset, current_depth+1);

    shared_ptr<Node> new_decision = make_shared<Node>(
        Decision { 
            (size_t)ideal_variance.chosen_feature,
            ideal_variance.chosen_threshold,
            new_left,
            new_right
        }
    );
    return new_decision;    
}

int main() {

    // double result = traverse(*head, features);
    // println("The result is {:g}", result);

  //  unique_ptr<Dataset> dataset = make_unique<Dataset>("data/fragmented/1.csv"); 

    auto [dataset, time] = loki(
        [](){
        return make_shared<Dataset>("data/fragmented/1.csv"); 
        }
    );
    println("Loading one dataset [:d ms]", time);
    
    dataset->display_1(43571); 

    dataset->show_feature_names();
    // dataset->label_by_feature("Class");


    shared_ptr<Dataset> test_dataset = make_shared<Dataset>("data/fragmented/2.csv");
    println("Loaded test dataset");
    println("Test dataset has {:d} datapoints", test_dataset->dp_count);
    println("OG dataset has {:d} ({:d}) datapoints", dataset->dp_count, dataset->features.at(0).size());

    dataset->combine(*test_dataset);
    println("Combined datasets");
    println("{:d}", dataset->dp_count);
    dataset->display_1(78376);

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

    shared_ptr<Node> tree = build_tree(*dataset);
    GradientBoosterMetadata md{0.01, 10, 3, 100};
    GradientBooster gb(md, dataset);
    gb.generate();


    // int class_index = dataset->feature_index_to_name.at("Class");
    // println("Class index: {:d}", class_index);
    // for (int i = 0; i < dataset->dp_count; i++) {
    //     if (dataset->features.at(class_index).at(i) != dataset->labels.at(i)) {
    //         println("Wrong");
    //     }
    // }
}

//int 
    // IdealVariance thebest = findIdealVariance(*dataset);
