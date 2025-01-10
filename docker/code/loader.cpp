#include <torch/script.h> // One-stop header.
#include <torch/data.h>

#include <iostream>
#include <memory>
#include <string>
// int test(::DataLoader& data_loader){

  
// }

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-data>\n";
    return -1;
  }


  torch::jit::script::Module module;
  std::string data_path = argv[2];
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  torch::NoGradGuard no_grad;
  module.eval();
  auto test_dataset = torch::data::datasets::MNIST(
                        data_path, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());


  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), 64);

  double test_loss = 0;
  int32_t correct = 0;
  size_t dataset_size = 10000;
  size_t counter = 0;
  for (const auto& batch : *test_loader) {
    auto data = batch.data.to("cpu"), targets = batch.target.to("cpu");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(data);

    auto output = module.forward(inputs).toTensor();

    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    test_loss += torch::nll_loss(
                    output,
                    targets,
                    /*weight=*/{},
                    torch::Reduction::Sum)
                    .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    counter++;

  }

  test_loss /= dataset_size;
  std::printf("Accuracy: %.3f\n", static_cast<double>(correct) / dataset_size);

}