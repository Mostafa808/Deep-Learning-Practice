#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <thread>
#include <time.h>
#include <cassert>

float precision; int size_type;
decltype(size_type) cBias = 1, gPrevious = -1, gNext = 1, gFirst = 0;
decltype(precision) gAlpha = static_cast<decltype(precision)>(1e-3), gBeta = static_cast <decltype(precision)>(0.7),
gEpsilon = static_cast<decltype(precision)>(1e-8), gBeta_m = static_cast <decltype(precision)>(0.9),
gBeta_v = static_cast <decltype(precision)>(0.99),
gTinyNeg = static_cast <decltype(precision)>(-1e-3), gTinyPos = static_cast <decltype(precision)>(1e-3);
std::vector<decltype(size_type)> gTraining_settings({ 1,1 });
std::vector<bool> gUpdateSettings({ 0,0,1 });

decltype(precision) randomRange(decltype(precision) max = 1, decltype(precision) min = 0, bool to_int = 0);
struct network;
struct layer;
struct node;
struct link;

struct network {
	decltype(size_type) size, tLayerSize = 5;
	std::vector <layer> layers;
	std::vector<decltype(precision)> teacherResults, errors, errorsGrid;
	decltype(precision) totalError = 0;
	int64_t epoch = 0;
	network(const std::vector<layer>& tLayers, decltype(size_type) tSize = 0);
	void feedforward(decltype(size_type) firstLayer, decltype(size_type) lastLayer);
	void backPropagation(const std::vector<bool>& updateSettings, decltype(size_type) firstLayer, decltype(size_type) lastLayer);
	void rescaleLinks();
	void rescaleTeacher(decltype(size_type) lastLayer = 0);
	void resetErrorsGrid();
	void resetTotalError();
	void normalized_init();
	void deep_learning_pretraining(std::vector<decltype(precision)> inputs, decltype(size_type) SingleSessionLength, decltype(size_type) pretrainingType = 1);
	void lossFunction(decltype(size_type) lastLayer = 0, decltype(size_type) formulaCode = 0, decltype(size_type) learnRange = -1);
	void setInputs(const std::vector<decltype(precision)>& input, decltype(size_type) layerIndex = 0);
	void getOutputs(std::vector<decltype(precision)>& output, decltype(size_type) layerIndex = 0);
	void trainingSession(const std::vector< std::vector<decltype(precision)>>& inputs, const std::vector< std::vector<decltype(precision)>>& tTeacherResults, const std::vector<decltype(size_type)>& settings, const std::vector<bool>& updateSettings, decltype(size_type) sessionLength, decltype(size_type) firstLayer = 0, decltype(size_type) lastLayer = 0, decltype(size_type) rolls = 1);

	void assignOutput(layer& outLayer);
	void dynamicSingleTraining(const std::vector<decltype(precision)>& input, const std::vector<decltype(precision)>& tTeacherResult, const std::vector<decltype(size_type)>& settings, const std::vector<bool>& updateSettings, decltype(size_type) sessionLength, decltype(size_type) firstLayer = 0, decltype(size_type) lastLayer = 0, decltype(size_type) rolls = 1);

	void viewArrayValues(const std::vector<decltype(precision)> data);
};
struct layer {
	decltype(size_type) size = 1;
	std::vector <node> nodes;
	layer(const std::vector<node>& tNodes, decltype(size_type) tSize = 0);
	layer();
	void feedforward(const layer& previous);
	void OutputLayerBackPropagation(const std::vector<decltype(precision)>& errorsGrid, const layer& next, const std::vector<bool>& updateSettings);
	void backPropagation(const layer& next, const std::vector<bool>& updateSettings);
	void rescaleLinks(const layer& next);
	void normalized_init(decltype(size_type) inputsLen, decltype(size_type) outputsLen);
	void viewNodesValues();

};
struct node {
	decltype(precision) value = 1, min = -2, max = 2, preValue = 0;
	decltype(size_type) formulaCode = 1, momentumType = 1;
	std::vector<link> links;
	node(); node(std::vector<link> tLinks, decltype(precision) tMin = 0, decltype(precision) tMax = 1, decltype(size_type) tFormulaCode = 1, decltype(size_type) tMomentumType = 1);
	void feedforward(const layer& previous, decltype(size_type) linkIndex);
	void backPropagation(const std::vector<decltype(precision)>& outputGrid, const std::vector<bool>& updateSettings);
	void rescaleLinks(const layer& next);
	void normalized_init(decltype(size_type) inputsLen, decltype(size_type) outputsLen);
	static decltype(precision) activitation(decltype(precision) input, decltype(size_type) formulaCode = 0,
		decltype(precision) min = 0, decltype(precision) max = 1);
	static decltype(precision) activitationGrid(decltype(precision) input, decltype(size_type) formulaCode = 0,
		decltype(precision) min = 0, decltype(precision) max = 1);
	static decltype(precision) momentum(decltype(precision) oldGrid, decltype(size_type) momentumType = 0, decltype(precision) beta = gBeta);
};
struct link {
	decltype(precision) weight, grid, change = 0, epsilon = gEpsilon, eta = gAlpha,
		m = 0, v = 0, mVector = 0, vVector = 0;
	link(decltype(precision) tWeight, decltype(precision) tGrid);
	link(decltype(precision) tWeight);
	link();
};

network::network(const std::vector<layer>& tLayers, decltype(size_type) tSize) {
	if (tSize) {
		size = tSize;
	}
	else {
		size = tSize = static_cast<decltype(size_type)>(tLayers.size());
	}
	// The minimum size of network is two input and output.
	assert(size >= 2);
	std::vector<node> tLayerNodes;
	for (decltype(size_type) layerIndex = 0; layerIndex < size; layerIndex++) {
		if (layerIndex < tLayers.size()) {
			layers.push_back(tLayers[layerIndex]);
		}
		else {
			tLayerNodes.resize(tLayerSize);
			layers.push_back(layer(tLayerNodes));
		}

	}
	rescaleLinks(); rescaleTeacher();
}
void network::feedforward(decltype(size_type) firstLayer, decltype(size_type) lastLayer) {
	for (decltype(size_type) layerIndex = firstLayer + gNext; layerIndex < lastLayer + gNext; layerIndex++) {
		layers[layerIndex].feedforward(layers[layerIndex + gPrevious]);
	}
	epoch++;
}
void network::backPropagation(const std::vector<bool>& updateSettings, decltype(size_type) firstLayer, decltype(size_type) lastLayer) {
	layers[lastLayer + gPrevious].OutputLayerBackPropagation(errorsGrid, layers[lastLayer], updateSettings);
	for (decltype(size_type) layerIndex = (lastLayer + 2 * gPrevious); layerIndex >= firstLayer; layerIndex--) {
		layers[layerIndex].backPropagation(layers[layerIndex + gNext], updateSettings);
	}
	resetErrorsGrid();
}
void network::rescaleLinks() {
	for (decltype(size_type) layerIndex = 0; layerIndex < (size + gPrevious); layerIndex++) {
		layers[layerIndex].rescaleLinks(layers[layerIndex + gNext]);
	}
}
void network::rescaleTeacher(decltype(size_type) lastLayer) {
	if (!lastLayer) {
		lastLayer = size + gPrevious;
	}
	teacherResults.resize(layers[lastLayer].size, 0);
	errors.resize(layers[lastLayer].size, 0);
	errorsGrid.resize(layers[lastLayer].size, 0);
}
void network::resetErrorsGrid() {
	errors.assign(errors.size(), 0);
	errorsGrid.assign(errorsGrid.size(), 0);
}
void network::resetTotalError() {
	totalError = 0; epoch = 0;
}
void network::normalized_init() {
	for (decltype(size_type) layerIndex = 0; layerIndex < (size + gPrevious); layerIndex++) {
		layers[layerIndex].normalized_init(static_cast<decltype(size_type)>(layers[gFirst].nodes.size()), static_cast<decltype(size_type)>(layers[size + gPrevious].nodes.size()));
	}
}
void network::deep_learning_pretraining(std::vector<decltype(precision)> inputs, decltype(size_type) SingleSessionLength, decltype(size_type) pretrainingType) {
	decltype(size_type) firstLayer = 0, lastLayer = firstLayer + 2 * gNext;
	switch (pretrainingType)
	{
	case 1:
		// Autoencoder method
		while (lastLayer < size) {
			rescaleTeacher(lastLayer);
			dynamicSingleTraining(inputs, inputs, gTraining_settings, gUpdateSettings, SingleSessionLength, firstLayer, lastLayer);
			firstLayer++, lastLayer++;
			getOutputs(inputs, firstLayer);
		}
		break;
	}
}
void network::lossFunction(decltype(size_type) lastLayer, decltype(size_type) formulaCode, decltype(size_type) learnRange) {
	if (!lastLayer) {
		lastLayer = size + gPrevious;
	}
	if (learnRange == -1) {
		learnRange = layers[lastLayer].size;
	}
	switch (formulaCode) {
	case 1:
		// E = (1/2) * sum((desiredOutput - nodeOutput)^2)
		// the total sum of the error in every node and in every training pattern
		for (decltype(size_type) nodeIndex = 0; nodeIndex < learnRange; nodeIndex++) {
			decltype(precision) tError = static_cast<decltype(precision)>(
				pow(teacherResults[nodeIndex] - layers[lastLayer].nodes[nodeIndex].value, 2) / 2
				);
			errors[nodeIndex] += tError;
			errorsGrid[nodeIndex] += layers[lastLayer].nodes[nodeIndex].value - teacherResults[nodeIndex];
			totalError += tError;
		}
		break;
	case 2:
		// Softmax cross entropy error
		// E = desiredOutput * ln(soft_max(nodeOutput))
		// soft_max = exp(nodeOutput) / sum(exp(nodesOutput))
		// the grid = (desiredOutput - soft_max(nodeOutput)) * nodeOutput
		decltype(precision) total_exp_nodes = 0, softmax_prob = 0;
		for (decltype(size_type) nodeIndex = 0; nodeIndex < layers[size + gPrevious].size; nodeIndex++) {
			total_exp_nodes += exp(layers[size + gPrevious].nodes[nodeIndex].value);
		}
		for (decltype(size_type) nodeIndex = 0; nodeIndex < layers[size + gPrevious].size; nodeIndex++) {
			// Compute the softmax probability for this node
			softmax_prob = exp(layers[size + gPrevious].nodes[nodeIndex].value) / total_exp_nodes;
			// Compute the cross-entropy loss for this node
			decltype(precision) tError = -teacherResults[nodeIndex] * log(softmax_prob);
			// Update errors and errorsGrid
			errors[nodeIndex] += tError;
			errorsGrid[nodeIndex] += (softmax_prob - teacherResults[nodeIndex]);
			totalError += tError;
		}
		break;
	}
}
void network::setInputs(const std::vector<decltype(precision)>& input, decltype(size_type) layerIndex) {
	assert(input.size() == layers[layerIndex].size);
	for (decltype(size_type) nodeIndex = 0; nodeIndex < layers[gFirst].size; nodeIndex++) {
		layers[layerIndex].nodes[nodeIndex].value = input[nodeIndex];
	}
}
void network::getOutputs(std::vector<decltype(precision)>& output, decltype(size_type) layerIndex) {
	output.resize(layers[layerIndex].size);
	for (decltype(size_type) nodeIndex = 0; nodeIndex < layers[gFirst].size; nodeIndex++) {
		output[nodeIndex] = layers[layerIndex].nodes[nodeIndex].value;
	}
}
void network::trainingSession(const std::vector< std::vector<decltype(precision)>>& inputs, const std::vector< std::vector<decltype(precision)>>& tTeacherResults, const std::vector<decltype(size_type)>& settings, const std::vector<bool>& updateSettings, decltype(size_type) sessionLength, decltype(size_type) firstLayer, decltype(size_type) lastLayer, decltype(size_type) rolls) {
	assert(inputs.size() == tTeacherResults.size());
	if (!lastLayer) {
		lastLayer = size + gPrevious;
	}
	decltype(size_type) learnRange;
	decltype(size_type) dataIndex = 0;
	for (decltype(size_type) session = 0; session < sessionLength; session++) {
		for (decltype(size_type) rollNum = 0; rollNum < rolls; rollNum++) {
			switch (settings[1]) // todo changing this to struct type.
			{
			case 0:
				// random selection
				dataIndex = static_cast<decltype(size_type)>(randomRange(static_cast<decltype(precision)>(inputs.size() - 1), 0e0, true));
				break;
			case 1:
				// sequent selection
				dataIndex = session % inputs.size();
				break;
			case 2:
				// constructive, random selection
				dataIndex = static_cast<decltype(size_type)>(randomRange(static_cast<decltype(precision)>(
					session / (sessionLength / inputs.size())
					), 0e0, true));
				break;
			case 3:
				// constructive, sequent selection
				dataIndex = session % (session / (sessionLength / inputs.size()) + gNext);
				break;
			}
			setInputs(inputs[dataIndex], firstLayer);
			if (teacherResults.size() > tTeacherResults[dataIndex].size()) {
				std::copy(tTeacherResults[dataIndex].begin(), tTeacherResults[dataIndex].end(), teacherResults.begin());
				learnRange = static_cast<decltype(size_type)>(tTeacherResults[dataIndex].size());
			}
			else {
				std::copy(tTeacherResults[dataIndex].begin(), tTeacherResults[dataIndex].begin() + teacherResults.size(), teacherResults.begin());
				learnRange = static_cast<decltype(size_type)>(teacherResults.size());
			}
			feedforward(firstLayer, lastLayer);
			lossFunction(lastLayer, settings[0], static_cast<decltype(size_type)>(tTeacherResults[dataIndex].size()));
			// debuging section
			std::cout << "Input Values are: "; layers[firstLayer].viewNodesValues(); std::cout << std::endl;
			std::cout << "Output Values are: "; layers[lastLayer].viewNodesValues(); std::cout << std::endl;
			std::cout << "Desired Output Values are: "; viewArrayValues(teacherResults); std::cout << std::endl;
			std::cout << "Output Errors Values are: "; viewArrayValues(errors); std::cout << std::endl;
			std::cout << "Errors Grid Values are: "; viewArrayValues(errorsGrid); std::cout << std::endl;
			std::cout << "Total Error: " << totalError << std::endl;
			std::cout << "Average Error: " << static_cast<decltype(precision)>(totalError) / epoch << std::endl << std::endl;
			// end of debug
		}
		backPropagation(updateSettings, firstLayer, lastLayer);
	}
}
void network::dynamicSingleTraining(const std::vector<decltype(precision)>& input, const std::vector<decltype(precision)>& tTeacherResult, const std::vector<decltype(size_type)>& settings, const std::vector<bool>& updateSettings, decltype(size_type) sessionLength, decltype(size_type) firstLayer, decltype(size_type) lastLayer, decltype(size_type) rolls) {
	if (!lastLayer) {
		lastLayer = size + gPrevious;
	}
	decltype(size_type) learnRange;
	for (decltype(size_type) session = 0; session < sessionLength; session++) {
		for (decltype(size_type) rollNum = 0; rollNum < rolls; rollNum++) {
			setInputs(input, firstLayer);
			if (teacherResults.size() > tTeacherResult.size()) {
				std::copy(tTeacherResult.begin(), tTeacherResult.end(), teacherResults.begin());
				learnRange = static_cast<decltype(size_type)>(tTeacherResult.size());
			}
			else {
				std::copy(tTeacherResult.begin(), tTeacherResult.begin() + teacherResults.size(), teacherResults.begin());
				learnRange = static_cast<decltype(size_type)>(teacherResults.size());
			}
			feedforward(firstLayer, lastLayer);
			lossFunction(lastLayer, settings[0], learnRange);
			// debuging section
			std::cout << "Input Values are: "; layers[firstLayer].viewNodesValues(); std::cout << std::endl;
			std::cout << "Output Values are: "; layers[lastLayer].viewNodesValues(); std::cout << std::endl;
			std::cout << "Desired Output Values are: "; viewArrayValues(teacherResults); std::cout << std::endl;
			std::cout << "Output Errors Values are: "; viewArrayValues(errors); std::cout << std::endl;
			std::cout << "Errors Grid Values are: "; viewArrayValues(errorsGrid); std::cout << std::endl;
			std::cout << "Total Error: " << totalError << std::endl;
			std::cout << "Average Error: " << static_cast<decltype(precision)>(totalError) / epoch << std::endl << std::endl;
			// end of debug
		}
		backPropagation(updateSettings, firstLayer, lastLayer);
	}
}
void network::assignOutput(layer& outLayer) {
	layers.push_back(outLayer); size++;
	rescaleTeacher(); layers[layers.size() + 2 * gPrevious].rescaleLinks(layers[layers.size() + gPrevious]);
}

void network::viewArrayValues(const std::vector<decltype(precision)> data) {
	for (decltype(size_type) element = 0; element < data.size() + gPrevious; element++) {
		std::cout << data[element] << ", ";
	}
	std::cout << data[data.size() + gPrevious] << std::endl;
}

layer::layer(const std::vector<node>& tNodes, decltype(size_type) tSize) {
	if (tSize) {
		size = tSize;
	}
	else
	{
		size = tSize = static_cast<decltype(size_type)>(tNodes.size());
	}
	// the minimus size of the layer is one plus the bias
	assert(size >= 1);
	for (decltype(size_type) nodeIndex = 0; nodeIndex < (size + cBias); nodeIndex++) {
		if (nodeIndex < tNodes.size()) {
			nodes.push_back(tNodes[nodeIndex]);
		}
		else {
			nodes.push_back(node());

		}
	}
}
layer::layer() {
	nodes.resize(size + cBias);
}
void layer::feedforward(const layer& previous) {
	for (decltype(size_type) nodeIndex = 0; nodeIndex < size; nodeIndex++) {
		nodes[nodeIndex].feedforward(previous, nodeIndex);
	}

}
void layer::OutputLayerBackPropagation(const std::vector<decltype(precision)>& errorsGrid, const layer& next, const std::vector<bool>& updateSettings) {
	std::vector<decltype(precision)> outputGrid(errorsGrid.size());
	for (decltype(size_type) grid = 0; grid < next.size; grid++) {
		outputGrid[grid] = errorsGrid[grid] * node::activitationGrid(next.nodes[grid].preValue, next.nodes[grid].formulaCode, next.nodes[grid].min, next.nodes[grid].max);
	}
	for (decltype(size_type) nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
		nodes[nodeIndex].backPropagation(outputGrid, updateSettings);
	}
}
void layer::backPropagation(const layer& next, const std::vector<bool>& updateSettings) {
	std::vector<decltype(precision)> outputGrid(next.size);
	for (decltype(size_type) nodeIndex = 0; nodeIndex < next.size; nodeIndex++) {
		decltype(precision) nodeTotalGrid = 0;
		for (decltype(size_type) linkIndex = 0; linkIndex < next.nodes[nodeIndex].links.size(); linkIndex++) {
			nodeTotalGrid += next.nodes[nodeIndex].links[linkIndex].grid;
		}
		outputGrid[nodeIndex] = nodeTotalGrid * node::activitationGrid(next.nodes[nodeIndex].preValue, next.nodes[nodeIndex].formulaCode, next.nodes[nodeIndex].min, next.nodes[nodeIndex].max);
	}
	for (decltype(size_type) nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
		nodes[nodeIndex].backPropagation(outputGrid, updateSettings);
	}
}
void layer::rescaleLinks(const layer& next) {
	for (decltype(size_type) nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
		nodes[nodeIndex].rescaleLinks(next);
	}
}
void layer::normalized_init(decltype(size_type) inputsLen, decltype(size_type) outputsLen) {
	for (decltype(size_type) nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
		nodes[nodeIndex].normalized_init(inputsLen, outputsLen);
	}
}
void layer::viewNodesValues() {
	for (decltype(size_type) nodeIndex = 0; nodeIndex < nodes.size() + gPrevious; nodeIndex++) {
		std::cout << nodes[nodeIndex].value << ", ";
	}
	std::cout << nodes[nodes.size() + gPrevious].value << std::endl;
}
node::node() {}
node::node(std::vector<link> tLinks, decltype(precision) tMin, decltype(precision) tMax, decltype(size_type) tFormulaCode, decltype(size_type) tMomentumType) : links(tLinks), min(tMin), max(tMax), formulaCode(tFormulaCode), momentumType(tMomentumType) {}

void node::feedforward(const layer& previous, decltype(size_type) linkIndex) {
	preValue = 0;
	for (decltype(size_type) inputNode = 0; inputNode < previous.nodes.size(); inputNode++) {
		preValue += previous.nodes[inputNode].value * previous.nodes[inputNode].links[linkIndex].weight;
	}
	value = activitation(preValue, formulaCode, min, max);
}

void node::backPropagation(const std::vector<decltype(precision)>& outputGrid, const std::vector<bool>& updateSettings) {
	for (decltype(size_type) grid = 0; grid < outputGrid.size(); grid++) {
		if (updateSettings[2] == true) {
			// Adam update method
			links[grid].grid = -value * outputGrid[grid];
			links[grid].m = gBeta_m * links[grid].m + (1 - gBeta_m) * links[grid].grid;
			links[grid].v = gBeta_v * links[grid].v + (1 - gBeta_v) * static_cast<decltype(precision)>(pow(links[grid].grid, 2));
			links[grid].mVector = links[grid].m / (1 - gBeta_m);
			links[grid].vVector = links[grid].v / (1 - gBeta_v);
			links[grid].weight += gAlpha * links[grid].mVector / (gEpsilon + sqrt(links[grid].vVector)); // might alpha be eta and gEpsilon be epsilon for some reasons!

		}
		else {
			if (updateSettings[0] == true) {
				// add momentum (acceleration) factor
				links[grid].change = momentum(links[grid].grid, momentumType);
			}
			links[grid].grid = -value * outputGrid[grid];
			if (updateSettings[1] == true) {
				// adaptive learning speed (AdaGrad method)
				links[grid].epsilon += static_cast<decltype(precision)>(pow(links[grid].grid, 2));
				links[grid].eta = gAlpha / sqrt(links[grid].epsilon);
			}

			links[grid].change += links[grid].eta * links[grid].grid;
			links[grid].weight += links[grid].change;
		}
	}

}
void node::rescaleLinks(const layer& next) {
	links.resize(next.size);
}
void node::normalized_init(decltype(size_type) inputsLen, decltype(size_type) outputsLen) {
	for (decltype(size_type) weight = 0; weight < links.size(); weight++) {
		links[weight].weight = randomRange(static_cast<decltype(precision)>(-sqrt(6 * gNext) / sqrt(inputsLen + outputsLen)), static_cast<decltype(precision)>(sqrt(6 * gNext) / sqrt(inputsLen + outputsLen)));
	}
}
decltype(precision) node::activitation(decltype(precision) input, decltype(size_type) formulaCode,
	decltype(precision) min, decltype(precision) max) {
	switch (formulaCode) {
	case 1:
		// sigmoid function
		return static_cast<decltype(precision)>(min + ((max - min) / (1.0 + exp(-input))));
	case 2:
		// Rectified linear unit(ReLU) function
		if (input <= min) return min;
		else if (input >= max) return max;
		break;
	default:
		break;
	}
	return input;
}
decltype(precision) node::activitationGrid(decltype(precision) input, decltype(size_type) formulaCode,
	decltype(precision) min, decltype(precision) max) {
	switch (formulaCode) {
	case 1:
		// sigmoid function Grid
		// (max-min) * exp(-input) / (1.0 + exp(-input))^2
		return static_cast<decltype(precision)>((max - min) * exp(-input) / pow(1.0 + exp(-input), 2));
	case 2:
		// Rectified linear unit(ReLU) function Grid
		if (input < min || input > max) return 0;
		else return 1;
		break;
	default:
		break;
	}
	return 1;
}
decltype(precision) node::momentum(decltype(precision) oldGrid, decltype(size_type) momentumType, decltype(precision) beta) {
	decltype(precision) result = 0;
	switch (momentumType) {
	case 1:
		result = oldGrid * beta;
		break;
	default:
		break;
	}
	return result;
}

link::link(decltype(precision) tWeight, decltype(precision) tGrid) : weight(tWeight), grid(tGrid) {}
link::link(decltype(precision) tWeight) : weight(tWeight), grid(0) {}
link::link() : weight(randomRange(static_cast<decltype(precision)>(0.1), static_cast<decltype(precision)>(-0.1))), grid(0) {}

decltype(precision) randomRange(decltype(precision) max, decltype(precision) min, bool to_int) {
	if (to_int) {
		max += 1 + gTinyNeg;
	}
	return (static_cast<decltype(precision)>(rand()) / (RAND_MAX) * (max - min) + min);
}

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
		{-1,-1}, {1,-1}, {-1,1}, {1,1}
		});
	std::vector<std::vector<decltype(precision)>> outputs({
		{1,-1}, {-1,1}, {-1,1}, {1,-1}
		});
	snake.normalized_init();
	snake.deep_learning_pretraining(inputs[1], 300);
	snake.trainingSession(inputs, outputs, gTraining_settings, gUpdateSettings, 10000);

	return 0;
}