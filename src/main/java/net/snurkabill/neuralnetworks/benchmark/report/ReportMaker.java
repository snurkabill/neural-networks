package net.snurkabill.neuralnetworks.benchmark.report;

import net.snurkabill.neuralnetworks.results.ResultsSummary;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class ReportMaker {

    private static final int CHART_SIZE_X = 1280;
    private static final int CHART_SIZE_Y = 720;

    private static final Logger LOGGER = LoggerFactory.getLogger(ReportMaker.class);

    // TODO: report picker to pick anything from BasicTestResults
    public static void chart(final File target, final List<ResultsSummary> data, final String chartName,
                      final String labelX, final String labelY) {
        // prepare data
        final XYSeriesCollection dataset = new XYSeriesCollection();
        for (ResultsSummary summary : data) {
            final XYSeries xy = new XYSeries(summary.getName());
            long distributedSum = 0;
            for (int i = 0; i < summary.getResults().size(); i++) {
                distributedSum += summary.getResults().get(i).getNumOfTestedItems();
                xy.add(distributedSum, summary.getResults().get(i).getSuccess());
            }
            dataset.addSeries(xy);
        }
        // prepare axes
        final NumberAxis xAxis = new NumberAxis(labelX);
        final NumberAxis yAxis = new NumberAxis(labelY);
        // prepare chart
        final XYPlot plot = new XYPlot(dataset, xAxis, yAxis, new XYLineAndShapeRenderer(true, true));
        final JFreeChart chart = new JFreeChart(chartName, JFreeChart.DEFAULT_TITLE_FONT, plot, true);
        // draw chart
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