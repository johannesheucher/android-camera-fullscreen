package com.johannes.camerafullscreen;

import android.os.AsyncTask;
import android.widget.TextView;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class TemplateMatcherTask extends AsyncTask<Mat, Void, Double> {

    private Mat template;
    private Mat templateMask;
    private Mat sourceMask;
    private int matchFunction;
    private TextView resultLabel;

    public TemplateMatcherTask(Mat template, Mat templateMask, Mat sourceMask, int matchFunction, TextView resultLabel) {
        this.template = template;
        this.templateMask = templateMask;
        this.sourceMask = sourceMask;
        this.matchFunction = matchFunction;
        this.resultLabel = resultLabel;
    }


    @Override
    protected Double doInBackground(Mat... params) {
        Mat image = params[0];

        // turn into gray edge image
        //Imgproc.cvtColor(matRgba, matGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.Canny(image, image, 50, 150);

        // apply mask
        Mat matGrayMasked = new Mat(image.cols(), image.rows(), CvType.CV_8UC1);
        image.copyTo(matGrayMasked, sourceMask);
        image = matGrayMasked;


        // blur edge image and binarize afterwards
        Imgproc.blur(image, image, new Size(20, 20));
        Imgproc.threshold(image, image, 15, 255, Imgproc.THRESH_BINARY);

        Mat matchResult = new Mat(image.rows() - template.rows() + 1, image.cols() - template.cols() + 1, CvType.CV_32FC1);
        Imgproc.matchTemplate(image, template, matchResult, matchFunction, templateMask);


        Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(matchResult);
        if (minValueFunction(matchFunction)) {
            return minMaxResult.minVal;
        } else {
            return minMaxResult.maxVal;
        }
    }


    public static boolean minValueFunction(int matchFunction) {
        return (matchFunction == Imgproc.TM_SQDIFF || matchFunction == Imgproc.TM_SQDIFF_NORMED);
    }


    @Override
    protected void onPostExecute(Double aDouble) {
        super.onPostExecute(aDouble);
        resultLabel.setText(aDouble.toString());
    }
}
