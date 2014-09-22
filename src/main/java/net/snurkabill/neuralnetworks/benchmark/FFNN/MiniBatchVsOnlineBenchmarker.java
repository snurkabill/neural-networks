package net.snurkabill.neuralnetworks.benchmark.FFNN;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import net.snurkabill.neuralnetworks.benchmark.report.ReportMaker;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;
import net.snurkabill.neuralnetworks.results.ResultsSummary;

public class MiniBatchVsOnlineBenchmarker extends FeedForwardNetworkBatchBenchmarker {

	private final FeedForwardNetworkOfflineManager miniBatchManager;
	private final List<ResultsSummary> miniBatchSummary;
	
	public MiniBatchVsOnlineBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns, 
			int sizeOfTrainingBatch, int sizeOfMiniBatch, FeedForwardNetworkOfflineManager miniBatchNetworks,
            int pretrainingPercentageTrainingSet) {
		super(feedForwardNetworks, numOfRuns, sizeOfTrainingBatch, sizeOfMiniBatch, pretrainingPercentageTrainingSet);
		this.miniBatchManager = miniBatchNetworks;
		
		miniBatchSummary = new ArrayList<>();
		
		for (int i = 0; i < miniBatchManager.getNumOfNeuralNetworks(); i++) {
			miniBatchSummary.add(new ResultsSummary(miniBatchManager.getNetworkList().get(i).getName()));
		}
	}

	@Override
	public void benchmark() {
        if(this.pretrainingTrainingSet != 0) {
            feedForwardNetworks.pretrainInputNeurons(pretrainingTrainingSet);
        }
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
		concat();
		makeReport();
	}
	
	private void concat() {
		super.summary.addAll(this.miniBatchSummary);
	}

    @Override
    protected void makeReport() {
        ReportMaker.chart(new File("results.png"), summary, "FFNN", "iterations", "success");
    }
}
