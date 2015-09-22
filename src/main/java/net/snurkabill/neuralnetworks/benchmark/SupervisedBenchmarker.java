package net.snurkabill.neuralnetworks.benchmark;

import net.snurkabill.neuralnetworks.benchmark.report.ReportMaker;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;

import java.io.File;

public class SupervisedBenchmarker {

    private final MasterNetworkManager manager;
    private final int numOfRuns;
    private final int sizeOfTrainingBatch;

    private final String path;

    public SupervisedBenchmarker(int numOfRuns, int sizeOfTrainingBatch, MasterNetworkManager manager, String path) {
        this.numOfRuns = numOfRuns;
        this.sizeOfTrainingBatch = sizeOfTrainingBatch;
        this.manager = manager;
        this.path = path;
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
        File dir = new File(path);
        if (dir.exists()) {
            dir.delete();
        }
        new File(path).mkdir();
        // TODO: move picking into contructor
        ReportMaker.chart(new File(path + "/time_success.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_TIME, ReportMaker.Picking.PICK_SUCCESS);
        ReportMaker.chart(new File(path + "/iterations_success.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_ITERATIONS, ReportMaker.Picking.PICK_SUCCESS);
        ReportMaker.chart(new File(path + "/time_error.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_TIME, ReportMaker.Picking.PICK_ERROR);
        ReportMaker.chart(new File(path + "/iterations_error.png"), manager.getResults(), manager.getNameOfProblem(),
                ReportMaker.SortBy.BY_ITERATIONS, ReportMaker.Picking.PICK_ERROR);
    }
}
