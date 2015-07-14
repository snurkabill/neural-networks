package net.snurkabill.neuralnetworks.scenario;

import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;

public class LogicFunctions {

    protected NewDataItem[] createNotAnd() {
        NewDataItem[] array = new NewDataItem[3];
        array[0] = new NewDataItem(new double[]{0.0, 1.0}, new double[]{0, 1});
        array[1] = new NewDataItem(new double[]{1.0, 0.0}, new double[]{0, 1});
        array[2] = new NewDataItem(new double[]{0.0, 0.0}, new double[]{0, 1});
        return array;
    }

    protected NewDataItem[] createAnd() {
        NewDataItem[] array = new NewDataItem[3];
        array[0] = new NewDataItem(new double[]{1.0, 1.0}, new double[]{1, 0});
        array[1] = new NewDataItem(new double[]{1.0, 1.0}, new double[]{1, 0});
        array[2] = new NewDataItem(new double[]{1.0, 1.0}, new double[]{1, 0});
        return array;
    }

    protected NewDataItem[] createOr() {
        NewDataItem[] array = new NewDataItem[3];
        array[0] = new NewDataItem(new double[]{1.0, 1.0}, new double[]{1, 0});
        array[1] = new NewDataItem(new double[]{0.0, 1.0}, new double[]{1, 0});
        array[2] = new NewDataItem(new double[]{1.0, 0.0}, new double[]{1, 0});
        return array;
    }

    protected NewDataItem[] createNotOr() {
        NewDataItem[] array = new NewDataItem[3];
        array[0] = new NewDataItem(new double[]{0.0, 0.0}, new double[]{0, 1});
        array[1] = new NewDataItem(new double[]{0.0, 0.0}, new double[]{0, 1});
        array[2] = new NewDataItem(new double[]{0.0, 0.0}, new double[]{0, 1});
        return array;
    }

    protected NewDataItem[] createXor() {
        NewDataItem[] array = new NewDataItem[2];
        array[0] = new NewDataItem(new double[]{0.0, 1.0}, new double[]{1, 0});
        array[1] = new NewDataItem(new double[]{1.0, 0.0}, new double[]{1, 0});
        return array;
    }

    protected NewDataItem[] createNotXor() {
        NewDataItem[] array = new NewDataItem[2];
        array[0] = new NewDataItem(new double[]{0.0, 0.0}, new double[]{0, 1});
        array[1] = new NewDataItem(new double[]{1.0, 1.0}, new double[]{0, 1});
        return array;
    }

}
