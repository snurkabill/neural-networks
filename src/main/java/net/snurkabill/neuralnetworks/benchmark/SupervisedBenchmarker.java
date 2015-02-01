package net.snurkabill.neuralnetworks.benchmark;

import net.snurkabill.neuralnetworks.benchmark.report.ReportMaker;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;

import java.io.File;

public class SupervisedBenchmarker {

    private final MasterNetworkManager manager;
    private final int numOfRuns;
    private final int sizeOfTrainingBatch;

    public SupervisedBenchmarker(int numOfRuns, int sizeOfTrainingBatch, MasterNetworkManager manager) {
        this.numOfRuns = numOfRuns;
        this.sizeOfTrainingBatch = sizeOfTrainingBatch;
        this.manager = manager;
    }

    public void benchmark() {
        manager.testNetworks();
        for (int i = 0; i < numOfRuns; i++) {
            manager.trainNetworks(sizeOfTrainingBatch);
            manager.testNetworks();
        }
        makeReport();
    }

    protected void makeReport() {
        // TODO: move picking into contructor
        ReportMaker.chart(new File("results_timeXsuccess.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_TIME, ReportMaker.Picking.PICK_SUCCESS);
        ReportMaker.chart(new File("results_iterationsXsuccess.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_ITERATIONS, ReportMaker.Picking.PICK_SUCCESS);
        ReportMaker.chart(new File("results_timeXerror.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_TIME, ReportMaker.Picking.PICK_ERROR);
        ReportMaker.chart(new File("results_iterationsXerror.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_ITERATIONS, ReportMaker.Picking.PICK_ERROR);
    }
}
