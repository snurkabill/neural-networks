package net.snurkabill.neuralnetworks.heuristic.calculators;

import net.snurkabill.neuralnetworks.heuristic.BasicHeuristic;
import net.snurkabill.neuralnetworks.results.NetworkResults;

import java.util.List;

public interface HeuristicCalculator {

    BasicHeuristic calculateNewHeuristic(List<NetworkResults> results);

    BasicHeuristic calculateNewHeuristic(List<NetworkResults> results, BasicHeuristic heuristic);
}
