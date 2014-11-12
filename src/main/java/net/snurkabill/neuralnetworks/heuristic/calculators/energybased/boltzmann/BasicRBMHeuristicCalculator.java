package net.snurkabill.neuralnetworks.heuristic.calculators.energybased.boltzmann;

import net.snurkabill.neuralnetworks.heuristic.Heuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.results.NetworkResults;

import java.util.List;

public class BasicRBMHeuristicCalculator implements HeuristicCalculator {


    @Override
    public Heuristic calculateNewHeuristic(List<NetworkResults> results) {
        NetworkResults lastOne = results.get(results.size() - 1);
        HeuristicRBM heuristic = HeuristicRBM.createStartingHeuristicParams();
        System.out.println("Successs: " + lastOne.getComparableSuccess());
        if (lastOne.getComparableSuccess() > 30) {
            heuristic.constructiveDivergenceIndex = 1;
            heuristic.numOfTrainingIterations = 5;
            heuristic.temperature = 1;
            heuristic.learningRate = 0.2;
            heuristic.momentum = 0.5;
        }
        if (lastOne.getComparableSuccess() > 40) {
            heuristic.constructiveDivergenceIndex = 1;
            heuristic.numOfTrainingIterations = 5;
            heuristic.temperature = 1;
            heuristic.learningRate = 0.2;
            heuristic.momentum = 0.5;
        }
        if (lastOne.getComparableSuccess() > 50) {
            heuristic.constructiveDivergenceIndex = 1;
            heuristic.numOfTrainingIterations = 10;
            heuristic.temperature = 1;
            heuristic.learningRate = 0.1;
            heuristic.momentum = 0.5;
        }
        if (lastOne.getComparableSuccess() > 60) {
            heuristic.constructiveDivergenceIndex = 1;
            heuristic.numOfTrainingIterations = 10;
            heuristic.learningRate = 0.1;
            heuristic.momentum = 0.3;
        }
        if (lastOne.getComparableSuccess() > 70) {
            heuristic.constructiveDivergenceIndex = 3;
            heuristic.numOfTrainingIterations = 10;
            heuristic.temperature = 1;
            heuristic.learningRate = 0.01;
            heuristic.momentum = 0.2;
        }
        if (lastOne.getComparableSuccess() > 80) {
            heuristic.constructiveDivergenceIndex = 3;
            heuristic.numOfTrainingIterations = 30;
            heuristic.temperature = 0.9;
            heuristic.learningRate = 0.01;
            heuristic.momentum = 0.1;
        }
        if (lastOne.getComparableSuccess() > 90) {
            heuristic.constructiveDivergenceIndex = 5;
            heuristic.numOfTrainingIterations = 30;
            heuristic.temperature = 0.8;
            heuristic.learningRate = 0.01;
            heuristic.momentum = 0.1;
        }
        return heuristic;
    }
}
