package net.snurkabill.neuralnetworks.scenario;

import net.snurkabill.neuralnetworks.data.database.DataItem;

import java.util.ArrayList;
import java.util.List;

public class LogicFunctions {

    protected List<DataItem> createNotAnd() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{0.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        return list;
    }

    protected List<DataItem> createAnd() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 1.0}));
        return list;
    }

    protected List<DataItem> createOr() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{0.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        return list;
    }

    protected List<DataItem> createNotOr() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{0.0, 0.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        return list;
    }

    protected List<DataItem> createXor() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{0.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        return list;
    }

    protected List<DataItem> createNotXor() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        return list;
    }

}
