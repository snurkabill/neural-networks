package net.snurkabill.neuralnetworks.benchmark.report;

import net.snurkabill.neuralnetworks.results.ResultsSummary;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class ReportMaker {

    private static final int CHART_SIZE_X = 1280;
    private static final int CHART_SIZE_Y = 720;

    private static final Logger LOGGER = LoggerFactory.getLogger("ReportMaker");

    public enum SortBy {
        BY_TIME("Milliseconds"),
        BY_ITERATIONS("Iterations");
        private final String string;
        private SortBy(String string) {
            this.string = string;
        }
        public String getString() {
            return string;
        }
    };

    public enum Picking {
        PICK_SUCCESS("Success"),
        PICK_ERROR("Error");
        private final String string;
        private Picking(String string) {
            this.string = string;
        }
        public String getString() {
            return string;
        }
    }

    // TODO: report picker to pick anything from BasicTestResults
    public static void chart(final File target, final List<ResultsSummary> data, final String chartName,
                             final SortBy sortby, Picking picking) {
        LOGGER.info("Preparing data for report");
        final XYSeriesCollection dataset = new XYSeriesCollection();
        for (ResultsSummary summary : data) {
            final XYSeries xy = new XYSeries(summary.getName());
            long distributedSum = 0;
            for (int i = 0; i < summary.getResults().size(); i++) {
                if(sortby == SortBy.BY_ITERATIONS) {
                    distributedSum += summary.getResults().get(i).getNumOfTrainedItems();
                } else if (sortby == SortBy.BY_TIME) {
                    distributedSum += summary.getResults().get(i).getMilliseconds();
                } else {
                    throw new IllegalArgumentException("Not possible");
                }
                if(picking == Picking.PICK_ERROR) {
                    xy.add(distributedSum, summary.getResults().get(i).getGlobalError());
                } else if (picking == Picking.PICK_SUCCESS) {
                    xy.add(distributedSum, summary.getResults().get(i).getSuccess());
                } else {
                    throw new IllegalArgumentException("Not possible");
                }
            }
            dataset.addSeries(xy);
        }
        LOGGER.info("Preparing axes");
        final NumberAxis xAxis = new NumberAxis(sortby.getString());
        final NumberAxis yAxis = new NumberAxis(picking.getString());
        LOGGER.info("Preparing chart");
        final XYPlot plot = new XYPlot(dataset, xAxis, yAxis, new XYLineAndShapeRenderer(true, true));
        plot.setBackgroundPaint(Color.DARK_GRAY);
        plot.setOutlinePaint(Color.DARK_GRAY);
        final JFreeChart chart = new JFreeChart(chartName, JFreeChart.DEFAULT_TITLE_FONT, plot, true);
        LOGGER.info("Drawing chart");
        try {
            if (target.exists()) {
                target.delete();
            }
            ChartUtilities.saveChartAsPNG(target, chart, ReportMaker.CHART_SIZE_X, ReportMaker.CHART_SIZE_Y, null);
            ReportMaker.LOGGER.info("Chart written as {}.", target.getAbsolutePath());
        } catch (final IOException e) {
            ReportMaker.LOGGER.warn("Cannot write chart file.", e);
        }
    }
}