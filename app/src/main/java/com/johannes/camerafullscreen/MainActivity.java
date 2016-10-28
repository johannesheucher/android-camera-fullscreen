package com.johannes.camerafullscreen;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, View.OnTouchListener {

    private CameraView cameraView;
    private TextView curValueText;
    private TextView maxValueText;
    private double maxValue;
    private Mat template = null;
    private Mat source = null;
    private Mat mask = null;
    private Mat sourceMask = null;
    private Mat matGray = null;
    private int matchFunction;

    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == BaseLoaderCallback.SUCCESS) {
                cameraView.enableView();
                cameraView.setOnTouchListener(MainActivity.this);
                try {
                    template = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_template);
                    template = template.t();

                    source = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_source);
                    /*
                    for a test, create an image CONTAINING the template and some other stuff around it
                        a. stuff out of bounding box
                        b. stuff inside bounding box but outside mask -> test with mask
                    Load this image and test template against it. Results should be best possible!!!
                    */

                    //template = Utils.loadResource(mAppContext, R.drawable.wimmel_template);
                    mask = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_template_mask);
                    sourceMask = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_source_mask);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else {
                super.onManagerConnected(status);
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraView = (CameraView)findViewById(R.id.camera_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);

        curValueText = (TextView)findViewById(R.id.curValue);
        maxValueText = (TextView)findViewById(R.id.maxValue);

        matchFunction = Imgproc.TM_CCORR_NORMED;
        onTouch(null, null);
    }


    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }


    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i("MainActivity", "OpenCV loaded successfully");
            loaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        } else {
            Log.i("MainActivity", "OpenCV did not load");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, loaderCallback);
        }
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
    }


    @Override
    public void onCameraViewStopped() {
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        matGray = inputFrame.gray();
//        Mat matRgb = new Mat(matRgba.rows(), matRgba.cols(), CvType.CV_8UC3);
//        Imgproc.cvtColor(matRgba, matRgb, Imgproc.COLOR_RGBA2RGB);

//        if (matGray == null) {
//            matGray = new Mat(matRgba.cols(), matRgba.rows(), CvType.CV_8UC1);
//        }

        // turn into gray edge image
        //Imgproc.cvtColor(matRgba, matGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.Canny(matGray, matGray, 50, 150);

        // apply mask
        Mat matGrayMasked = new Mat(matGray.cols(), matGray.rows(), CvType.CV_8UC1);
        matGray.copyTo(matGrayMasked, sourceMask);
        matGray = matGrayMasked;


        //Imgproc.threshold(matGray, matGray, 160, 255, Imgproc.THRESH_BINARY);

        // TODO: Blur edge image?
        Imgproc.blur(matGray, matGray, new Size(20, 20));
        Imgproc.threshold(matGray, matGray, 15, 255, Imgproc.THRESH_BINARY);

//        if (source.cols() != matRgba.cols() && source.rows() != matRgba.rows()) {
//            Imgproc.resize(source, source, new Size(matRgba.cols(), matRgba.rows()));
//        }

        Mat curMat = matGray;

        final double val;
        Core.MinMaxLocResult matchLocation;
        matchLocation = match(curMat, template, mask);

        if (matchLocation != null) {
            final Point loc;
            if (minValueFunction()) {
                loc = matchLocation.minLoc;
                val = matchLocation.minVal;
            } else {
                loc = matchLocation.maxLoc;
                val = matchLocation.maxVal;
            }
            Imgproc.rectangle(curMat, loc, new Point(loc.x + template.cols(), loc.y + template.rows()), new Scalar(220, 180, 255), 6);
        } else {
            val = matchShapes(curMat, template);
        }

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                String curValueString = new Double(val).toString();
                curValueText.setText(curValueString);
                if ( minValueFunction() && val < maxValue ||
                        !minValueFunction() && val > maxValue) {
                    maxValue = val;
                    maxValueText.setText(curValueString);
                }

                // show template
                // convert to bitmap:
                Bitmap bm = Bitmap.createBitmap(template.cols(), template.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(template, bm);

                // find the imageview and draw it
                ImageView iv = (ImageView)findViewById(R.id.result_view);
                iv.setImageBitmap(bm);
            }
        });

        return curMat;
    }


    protected Core.MinMaxLocResult match(Mat image, Mat template, Mat mask) {
        Mat matchResult = new Mat(image.rows() - template.rows() + 1, image.cols() - template.cols() + 1, CvType.CV_32FC1);


        Imgproc.matchTemplate(image, template, matchResult, matchFunction, mask);
        //Core.normalize(matchResult, matchResult, 0, 1, Core.NORM_MINMAX, -1, new Mat());

        // draw result
//        Mat ucharMat = new Mat(matchResult.cols(), matchResult.rows(), CvType.CV_8UC1);
//        // scale values from 0..1 to 0..255
//        matchResult.convertTo(ucharMat, CV_8UC1, 255, 0);
//
//        final Bitmap bmp = Bitmap.createBitmap(matchResult.cols(), matchResult.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(ucharMat, bmp);
//        runOnUiThread(new Runnable() {
//            @Override
//            public void run() {
//                ImageView imgView = (ImageView)findViewById(R.id.result_view);
//                imgView.setImageBitmap(bmp);
//            }
//        });


        Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(matchResult);
        Log.i("MainActivity", new Double(minMaxResult.maxVal).toString());
        if (true) {//(minMaxResult.minVal < 1.3E9) {
            return minMaxResult;
        } else {
            return null;
        }
    }


    protected double matchShapes(Mat image, Mat template) {
        double result = Imgproc.matchShapes(image, template, Imgproc.CV_CONTOURS_MATCH_I1, 0);
        return result;
    }


    public boolean minValueFunction() {
        return (matchFunction == Imgproc.TM_SQDIFF || matchFunction == Imgproc.TM_SQDIFF_NORMED);
    }


    @Override
    public boolean onTouch(View view, MotionEvent event) {
        if (minValueFunction()) {
            maxValue = Double.MAX_VALUE;
        } else {
            maxValue = 0;
        }
        return false;
    }
}
