#include <memory>
#include <print>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>


class Dataset {
    std::vector<std::vector<double>> data;
    int rows;
    int columns;
    std::string filename;
    public:
        Dataset(std::string filename = "test.csv", int rows = 3, int columns = 5): rows(rows), columns(columns), filename(filename) {
            std::println("Dataset created from {:s}",  filename);
            data.resize(rows);
            for (int i = 0; i < rows; i++)  {
                data[i].resize(columns);
                for (int j = 0; j < columns; j++)  {
                    data[i][j] = j;
                }
            }
            // data[0]  = {1, 2, 3,  4, 5};
            // data[1]  = {1, 2, 3,  4, 5};
            // data[2]  = {1, 2, 3,  4, 5};
        }

        double mean_col(int col_index) const {
            if (col_index >=  columns) return -1;
            double sum = 0;
            for (const std::vector<double>& row : data) {
                sum += row[col_index];
            }
            sum /= rows;
            return sum;
        }

        int numrows() const {
            return rows;
        }

        void addrow(std::vector<double>& newrow) {
            rows++;
            data.push_back(newrow);
        }

        void printdata() const {
            for (int  i = 0; i < rows; i++) {
                std::print("[");
                for (int  j = 0; j < (int) data[i].size(); j++) {
                    std::print("{:f},", data[i][j]);
                }
                std::println("]");
            }
        }

        void apply_lambda(auto nasty) {
            for (int  i = 0; i < rows; i++) {
                for (int  j = 0; j < (int) data[i].size(); j++) {
                    data[i][j] = nasty(data[i][j]);
                }
            }
        }
};

int add(int& a, int b) {
    a += 5;
    return a+b;
}

void next_procedure(auto lambda_function)  {
    lambda_function(1, 2);

}

class my_class {
    int presumably_private;
    public:
        int presumably_public;
        my_class(int presumably_private):   presumably_private(presumably_private) {
            std::println("Instantiated");
        }

        my_class(int  presumably_private, int presumably_public): presumably_private(presumably_private),presumably_public(presumably_public)  {
            std::println("Class members:  {:d},{:d}", presumably_private, presumably_public);
        }

        int  getit()  {
            return presumably_private;
        }
};

using MaybeDataset = std::variant<std::string, Dataset>;


MaybeDataset openfile(std::string filename) {
    if (filename.substr(filename.length()-3, 3) == "csv") {
        return Dataset(filename);
    }
    return "Error: not a CSV.";
}

namespace whatever {
    template<typename... Ts>
    struct overloaded: Ts...  { using Ts::operator()...;  };
}

int main()  {
    int a = 5;
    int  b  = 12;
    std::println("Result: {:d}", add(a, b));

    std::unordered_map<std::string, int> my_map;
    my_map["key"] = 15;
    std::println("At  key: {:d}", my_map.at("key"));
    //  my_map.erase("key");
    std::println("At  key: {:d}", my_map.at("key"));


    std::vector<int> my_vec;
    my_vec.push_back(12);

    std::println("The value: {:d}", my_vec.back());

    auto thinger = [&my_vec](int a, int b)  {
        std::println("{:d}", a+b); 
        std::println("Last element in vector  is {:d}",  my_vec.back());
        my_vec.push_back(69);
        std::println("Last element in vector (post  lambda change) is {:d}",  my_vec.back());
    };

    next_procedure(thinger);
    std::println("The last value of the  vector is still {:d}", my_vec.back());

    using Option = std::variant<int, std::string>;
    Option my_option = 14;
    my_option = "hello";

    std::visit(whatever::overloaded{
        [](int& thevalue) {  std::println("The int value was  {:d}",  thevalue); },
        [](std::string& stringvalue) { std::println("the string value is {:s}", stringvalue);  }
    }, my_option);

    auto vova  =  std::make_unique<my_class>(11, 12);
    
    std::println("the vova private member  is: {:d}", vova->getit());

    auto my_dataset = std::make_unique<Dataset>();
    for (int i = 0; i < my_dataset->numrows(); i++) {
        std::println("Column {:d} mean: {:f}",  i, my_dataset->mean_col(i));
    }

    MaybeDataset perhaps = openfile("helloworld.csv");

    std::visit(whatever::overloaded{
        [](Dataset& my_dataset_2) { 
            my_dataset_2.printdata();
            std::vector<double> newrow({5, 6, 7, 8});
            std::println("Adding");
            my_dataset_2.addrow(newrow);
            my_dataset_2.printdata();

            auto  nasty = [](double value) {  return value*2; };
            my_dataset_2.apply_lambda(nasty);
            std::println("Lambda'ing");
            my_dataset_2.printdata();

        },
        [](std::string& error) {
            std::println("Error  getting dataset: {:s}",  error);
        }
    }, perhaps);




    return 0;
}

