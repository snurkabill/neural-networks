package net.snurkabill.neuralnetworks.heuristic.calculators;

import net.snurkabill.neuralnetworks.heuristic.Heuristic;
import net.snurkabill.neuralnetworks.results.NetworkResults;

import java.util.List;

public interface HeuristicCalculator {

    public Heuristic calculateNewHeuristic(List<NetworkResults> results);
}
