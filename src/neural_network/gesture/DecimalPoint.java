package neural_network.gesture;
public class DecimalPoint {
    public double x;
    public double y;

    public DecimalPoint() {

    }

    public DecimalPoint(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double distance(DecimalPoint p2) {
        return Math.hypot(p2.x - x, p2.y - y);
    }
}
