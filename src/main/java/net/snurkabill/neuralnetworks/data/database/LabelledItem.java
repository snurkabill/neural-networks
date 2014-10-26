package net.snurkabill.neuralnetworks.data.database;

public class LabelledItem extends DataItem {

    public int _class;

    public LabelledItem(double[] data, int _class) {
        super(data);
        this._class = _class;
    }

    public LabelledItem(DataItem item, int _class) {
        super(item.data);
        this._class = _class;
    }
}
