package net.snurkabill.neuralnetworks.benchmark.FFNN;

import java.io.IOException;
import java.util.List;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;

public class FeedForwardNetworkBatchBenchmarker extends FeedForwardNetworkBenchmarker {

	protected final int sizeOfMiniBatch;
	
	public FeedForwardNetworkBatchBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns,
                                              int sizeOfTrainingBatch, int sizeOfMiniBatch, int pretrainingTrainingSet) {
		super(feedForwardNetworks, numOfRuns, sizeOfTrainingBatch, pretrainingTrainingSet);
		this.sizeOfMiniBatch = sizeOfMiniBatch;
	}

	@Override
	public void benchmark() {
        if(pretrainingTrainingSet != 0) {
            feedForwardNetworks.pretrainInputNeurons(pretrainingTrainingSet);
        }
		for (int i = 0; i < numOfRuns; i++) {
			feedForwardNetworks.trainNetwork(sizeOfTrainingBatch, sizeOfMiniBatch);
			List<BasicTestResults> tmp = feedForwardNetworks.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp.size(); j++) {
				summary.get(j).add(tmp.get(j));
			}
		}
        makeReport();
	}
}
