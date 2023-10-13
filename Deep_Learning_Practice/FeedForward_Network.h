#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <thread>
#include <time.h>
#include <cassert>

static float precision; static int size_type;
static decltype(size_type) cBias = 1, gPrevious = -1, gNext = 1, gFirst = 0;
static decltype(precision) gAlpha = static_cast<decltype(precision)>(1e-3), gBeta = static_cast <decltype(precision)>(0.7),
gEpsilon = static_cast<decltype(precision)>(1e-8), gBeta_m = static_cast <decltype(precision)>(0.9),
gBeta_v = static_cast <decltype(precision)>(0.99),
gTinyNeg = static_cast <decltype(precision)>(-1e-3), gTinyPos = static_cast <decltype(precision)>(1e-3);
static std::vector<decltype(size_type)> gTraining_settings({ 1,1 });
static std::vector<bool> gUpdateSettings({ 0,0,1 });

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
	static decltype(precision) activation(decltype(precision) input, decltype(size_type) formulaCode = 0,
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
