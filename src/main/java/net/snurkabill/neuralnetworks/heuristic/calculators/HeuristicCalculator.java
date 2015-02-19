package net.snurkabill.neuralnetworks.heuristic.calculators;

import net.snurkabill.neuralnetworks.heuristic.BasicHeuristic;
import net.snurkabill.neuralnetworks.results.NetworkResults;

import java.util.List;

public interface HeuristicCalculator {

    public BasicHeuristic calculateNewHeuristic(List<NetworkResults> results);
}
