#include "FeedForward_Network.h"
int __cdecl main() {
	srand(static_cast<unsigned int>(time(0)));
	std::vector<link> links(5);
	std::vector<node> values(3);
	//std::cout << std::setprecision(6) << std::showpoint;

	layer input(values, 2);
	values.resize(0);
	layer middile(values, 30);
	layer output(values, 2);
	std::vector<layer> tLayers; tLayers.push_back(input);
	for (decltype(size_type) middileNum = 0; middileNum < 10; middileNum++) {
		tLayers.push_back(middile);
	}
	network snake(tLayers);
	snake.assignOutput(output);
	std::vector<std::vector<decltype(precision)>> inputs({
		{-1,-1}, {1,-1}, {-1,1}, {1,1} // xor training
		});
	std::vector<std::vector<decltype(precision)>> outputs({
		{1,-1}, {-1,1}, {-1,1}, {1,-1} // xor training
		});
	snake.normalized_init();
	snake.deep_learning_pretraining(inputs[1], 300);
	snake.trainingSession(inputs, outputs, gTraining_settings, gUpdateSettings, 10000);

	return 0;
}