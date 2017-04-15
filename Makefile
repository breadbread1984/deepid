CUDA_PREFIX=/usr
CAFFE_PREFIX=/home/xieyi/opt/caffe
CXXFLAGS=-std=c++14 `pkg-config --cflags opencv eigen3 dlib-1` -I${CUDA_PREFIX}/include -O2 \
-Isrc -I${CAFFE_PREFIX}/include -I${CAFFE_PREFIX}/.build_release/src/ 
LIBS=-std=c++14 `pkg-config --libs opencv eigen3 dlib-1` \
-L${CAFFE_PREFIX}/build/lib ${CAFFE_PREFIX}/.build_release/src/caffe/proto/caffe.pb.o -lcaffe -lglog -lprotobuf \
-lboost_serialization -lboost_filesystem -lboost_system -lboost_program_options -lboost_thread -lboost_regex -pthread -llapack
OBJS=$(patsubst src/%.cpp,src/%.o,$(wildcard src/*.cpp))

all: generate_training_samples convert move_training_results generate_training_samples2 roc transform main

generate_training_samples: src/generate_training_samples.o
	$(CXX) $^ $(LIBS) -o ${@}
	
convert: src/convert.o
	$(CXX) $^ $(LIBS) -o ${@}
	
move_training_results: src/move_training_results.o
	$(CXX) $^ $(LIBS) -o ${@}

generate_training_samples2: src/DeepID.o src/FeatureExtractor.o src/generate_training_samples2.o
	$(CXX) $^ $(LIBS) -o ${@}
	
roc: src/roc.o src/DeepID.o src/FeatureExtractor.o
	$(CXX) $^ $(LIBS) -o ${@}

transform: src/transform.o src/CCIPCA.o
	$(CXX) $^ $(LIBS) -o ${@}
	
main: src/main.o src/DeepID.o src/FeatureExtractor.o src/FaceRecognizer.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) generate_training_samples convert move_training_results generate_training_samples2 roc transform main $(OBJS)
