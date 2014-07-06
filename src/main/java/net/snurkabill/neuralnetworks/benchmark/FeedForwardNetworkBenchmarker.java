package net.snurkabill.neuralnetworks.benchmark;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.results.BasicTestResults;
import net.snurkabill.neuralnetworks.results.ResultsSummary;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;

public class FeedForwardNetworkBenchmarker implements Benchmarker {

	FeedForwardNetworkOfflineManager feedForwardNetworks;
	
	private final int numOfRuns;
	private final int sizeOfTrainingBatch;
	private final List<ResultsSummary> summary;
	
	public FeedForwardNetworkBenchmarker(FeedForwardNetworkOfflineManager feedForwardNetworks, int numOfRuns, 
			int sizeOfTrainingBatch) {
		this.feedForwardNetworks = feedForwardNetworks;
		this.numOfRuns = numOfRuns;
		this.sizeOfTrainingBatch = sizeOfTrainingBatch;
		summary = new ArrayList<>();
		
		for (int i = 0; i < feedForwardNetworks.getNumOfNeuralNetworks(); i++) {
			summary.add(new ResultsSummary());
		}
	}
	
	@Override
	public void benchmark() {
		int percentageOfTrainingSet = 20;
		feedForwardNetworks.pretrainInputNeurons(percentageOfTrainingSet);
		for (int i = 0; i < numOfRuns; i++) {
			feedForwardNetworks.trainNetwork(sizeOfTrainingBatch);
			List<BasicTestResults> tmp = feedForwardNetworks.testNetworkOnWholeTestingDataset();
			for (int j = 0; j < tmp.size(); j++) {
				summary.get(j).add(tmp.get(j));
			}
		}
		try {
			makeReport();
		} catch (IOException ex) {
			throw new RuntimeException("Making report went wrong: " + ex);
		}
	}
	
	public List<ResultsSummary> getResults() {
		return this.summary;
	}
	
	private void makeReport() throws IOException {
		try (FileWriter w = new FileWriter((new Date(System.currentTimeMillis())).toString() + ".csv")) {
			w.append("Name: ");
			w.append(",");
			List<FeedForwardNeuralNetwork> networks = feedForwardNetworks.getNetworkList();
			for (int i = 0; i < feedForwardNetworks.getNumOfNeuralNetworks(); i++) {
				w.append(networks.get(i).getName());
				w.append(",");
			}
			w.append("\n");
			int exclusiveSum = this.sizeOfTrainingBatch;
			for (int i = 0; i < numOfRuns; i++) {
				w.append(String.valueOf(exclusiveSum));
				w.append(",");
				exclusiveSum += this.sizeOfTrainingBatch;
				for (int j = 0; j < feedForwardNetworks.getNumOfNeuralNetworks(); j++) {
					w.append(String.valueOf(((SupervisedTestResults) summary.get(j).getResults().get(i)).getSuccessPercentage()));
					w.append(",");
				}
				w.append("\n");
			}
			w.flush();
		}
	}
}
