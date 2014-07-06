package net.snurkabill.neuralnetworks;

import java.util.List;
import net.snurkabill.neuralnetworks.results.BasicTestResults;

public interface NeuralNetworkManager {
	
	public void pretrainInputNeurons(int percentageOfDataset);
	
	public List<BasicTestResults> testNetworkOnWholeTestingDataset();
	
	public void trainNetwork(int numOfIterations);
	
}
