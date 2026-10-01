import java.io.File;

public class TrainingData {
    
    private File audioFile;
    private float[] expectedOutput;

    public TrainingData(File audioFile, float[] expectedOutput) {

        this.audioFile = audioFile;
        this.expectedOutput = expectedOutput;

    }

    public File getAudioFile() {
        return audioFile;
    }

    public float getExpectedOutputAt(int index) {
        return expectedOutput[index];
    }

    public static File[] getBatchFrom(File directory) {

        return directory.listFiles();

    }

}