CXX      := clang++
CXXFLAGS := -std=c++23 -Wall -Wextra -Ishared
SHARED_SRC := $(wildcard shared/*.cpp)
APPS     := $(patsubst %.cpp, %, $(wildcard *.cpp))

# Default: build everything
all: $(APPS)

# Pattern rule: Link each app with shared code
%: %.cpp $(SHARED_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(APPS)