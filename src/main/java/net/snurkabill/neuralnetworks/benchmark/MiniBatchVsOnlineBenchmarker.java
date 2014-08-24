package net.snurkabill.neuralnetworks.benchmark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;
import net.snurkabill.neuralnetworks.results.ResultsSummary;

public class MiniBatchVsOnlineBenchmarker extends FeedForwardNetowkrBatchBenchmarker {

	private final FeedForwardNetworkOfflineManager miniBatchManager;
	private final List<ResultsSummary> miniBatchSummary;
	
	public MiniBatchVsOnlineBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns, 
			int sizeOfTrainingBatch, int sizeOfMiniBatch, FeedForwardNetworkOfflineManager miniBatchNetworks) {
		super(feedForwardNetworks, numOfRuns, sizeOfTrainingBatch, sizeOfMiniBatch);
		this.miniBatchManager = miniBatchNetworks;
		
		miniBatchSummary = new ArrayList<>();
		
		for (int i = 0; i < miniBatchManager.getNumOfNeuralNetworks(); i++) {
			miniBatchSummary.add(new ResultsSummary());
		}
	}

	@Override
	public void benchmark() {
		int percentageOfTrainingSet = 20;
		feedForwardNetworks.pretrainInputNeurons(percentageOfTrainingSet);
		for (int i = 0; i < numOfRuns; i++) {
			feedForwardNetworks.trainNetwork(sizeOfTrainingBatch);
			miniBatchManager.trainNetwork(sizeOfTrainingBatch / sizeOfMiniBatch, sizeOfMiniBatch);
			List<BasicTestResults> tmp = feedForwardNetworks.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp.size(); j++) {
				summary.get(j).add(tmp.get(j));
			}
			List<BasicTestResults> tmp2 = miniBatchManager.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp2.size(); j++) {
				miniBatchSummary.get(j).add(tmp2.get(j));
			}
		}
		try {
			concat();
			makeReport();
		} catch (IOException ex) {
			throw new RuntimeException("Making report went wrong: " + ex);
		}	
	}
	
	private void concat() {
		super.summary.addAll(this.miniBatchSummary);
	}
}
