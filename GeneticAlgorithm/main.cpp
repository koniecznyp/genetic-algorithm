#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <bitset>
#include <math.h>
#include <thread>
#include <chrono>
#include <string>
#include <mutex>
#define POPULATION_SIZE 16
#define BITS 32									// chromosome size
#define P_m 0.1									// mutation probability
#define P_c 0.9									// crossover probability
#define LOOPS 5000								// iterations
#define THREADS 8
#define BLOCK_SIZE (POPULATION_SIZE / THREADS)	// data block size to calculate per thread

using namespace std;
using namespace chrono;

struct individual {
	bitset<BITS> chromosome;
	int adjustment;
	float adaptation;

	individual() {
		chromosome = bitset<BITS>(0);
		adjustment = 0;
		adaptation = 0.0f;
	}
};

individual individualAlpha;
individual bestIndividuals[THREADS];
individual population[POPULATION_SIZE];

int getRandomInteger(int rangeFrom, int rangeTo){
	random_device rseed;
	mt19937 rng(rseed());
	uniform_int<int> dist(rangeFrom, rangeTo);
	return dist(rng);
}

float getRandomFloat(float rangeFrom, float rangeTo){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(rangeFrom, rangeTo);
	return dis(gen);
}

int fitnessFuntion(int x){
	return pow(x, 3) + (2 * pow(x, 2)) + 1;
}

void saveBest(int threadId, int blockStart) {
	for (int i = 0; i < THREADS; ++i) {
		if (bestIndividuals[threadId].adjustment < population[blockStart + i].adjustment) {
			bestIndividuals[threadId] = population[blockStart + i];
		}
	}
}

void setParentalPool(int blockStart) {
	individual parentalPopulation[BLOCK_SIZE];
	for (int i = blockStart; i < blockStart + (BLOCK_SIZE); ++i) {
		float randomNumber = getRandomFloat(0.0, 1.0);
		float threshold = 0.0;
		for (int j = 0; j < BLOCK_SIZE; ++j){
			threshold += population[blockStart + j].adaptation;
			if (randomNumber < threshold) {
				parentalPopulation[j] = population[blockStart + j];
				break;
			}
		}
	}
	copy(begin(parentalPopulation), end(parentalPopulation), begin(population) + blockStart);
}

void crossover(int blockStart) {
	individual newPopulation[BLOCK_SIZE];
	for (int i = 0; i < BLOCK_SIZE; i += 2) {
		individual descendant1 = population[getRandomInteger(blockStart, (blockStart + BLOCK_SIZE) - 1)];
		individual descendant2 = population[getRandomInteger(blockStart, (blockStart + BLOCK_SIZE) - 1)];

		if (getRandomFloat(0.0, 1.0) < P_c) {
			int crossPoint = getRandomInteger(1, BITS - 2);
			for (int j = crossPoint; j >= 0; --j) {
				descendant1.chromosome[j] = descendant2.chromosome[j];
				descendant2.chromosome[j] = descendant1.chromosome[j];
			}
		}
		newPopulation[i] = descendant1;
		newPopulation[i + 1] = descendant2;
	}
	copy(begin(newPopulation), end(newPopulation), begin(population) + blockStart);
}

void mutation(int blockStart) {
	for (int i = 0; i < BLOCK_SIZE; ++i) {
		for (int j = 0; j < BITS; ++j) {
			if (getRandomFloat(0.0, 1.0) < P_m) {
				population[blockStart + i].chromosome[j] = !population[blockStart + i].chromosome[j];
			}
		}
	}
}

void calculateAdjustment(int blockStart) {
	int adjustmentSum = 0;
	for (int i = blockStart; i < blockStart + BLOCK_SIZE; ++i) {
		population[i].adjustment = fitnessFuntion(bitset<BITS>(population[i].chromosome).to_ulong());
		adjustmentSum += population[i].adjustment;
	}

	for (int i = blockStart; i < blockStart + BLOCK_SIZE; ++i) {
		population[i].adaptation = (float) population[i].adjustment / adjustmentSum;
	}
}

void algorithmLoop(int threadId) {
	int blockStart = threadId * BLOCK_SIZE;
	for (int i = 0; i < LOOPS; ++i) {
		calculateAdjustment(blockStart);
		setParentalPool(blockStart);
		crossover(blockStart);
		mutation(blockStart);
		saveBest(threadId, blockStart);
	}
}

void joinResults() {
	for (int i = 0; i < THREADS; ++i) {
		if (individualAlpha.adjustment < bestIndividuals[i].adjustment)
			individualAlpha = bestIndividuals[i];
	}
}

int main()
{
	vector<thread> threads;
	auto start = high_resolution_clock::now();

	for (int i = 0; i < POPULATION_SIZE; ++i)
		population[i].chromosome = bitset<BITS>(getRandomInteger(0, pow(BITS, 2) - 1));

	for (int i = 0; i < THREADS; ++i)
		threads.push_back(thread(algorithmLoop, i));

	for (auto& t : threads)
		t.join();

	joinResults();

	auto finish = high_resolution_clock::now();
	duration<double> elapsed = finish - start;
	printf("Elapsed time: %f [s]", elapsed.count());
	cin.get();
}