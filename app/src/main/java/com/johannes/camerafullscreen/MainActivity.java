package com.johannes.camerafullscreen;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
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
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, View.OnTouchListener {

    private CameraView cameraView;
    private TextView curValueText;
    private TextView maxValueText;
    private double maxValue = 0.0;
    private Mat template = null;
    private Mat matGray = null;

    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == BaseLoaderCallback.SUCCESS) {
                cameraView.enableView();
                cameraView.setOnTouchListener(MainActivity.this);
                try {
                    template = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_template);
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
        Mat matRgba = inputFrame.rgba();

        if (matGray == null) {
            matGray = new Mat(matRgba.cols(), matRgba.rows(), CvType.CV_8UC1);
        }

        Imgproc.cvtColor(matRgba, matGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.Canny(matGray, matGray, 50, 150);

        final Core.MinMaxLocResult matchLocation = match(matGray, template);

        if (matchLocation != null) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    String curValueString = new Double(matchLocation.maxVal).toString();
                    curValueText.setText(curValueString);
                    if (matchLocation.maxVal > maxValue) {
                        maxValue = matchLocation.maxVal;
                        maxValueText.setText(curValueString);
                    }
                }
            });
            Imgproc.rectangle(matRgba, matchLocation.maxLoc, new Point(matchLocation.maxLoc.x + template.cols(), matchLocation.maxLoc.y + template.rows()), new Scalar(50, 200, 255), 8);
        }

        return matRgba;
    }


    protected Core.MinMaxLocResult match(Mat image, Mat template) {
        Mat matchResult = new Mat(image.rows() - template.rows() + 1, image.cols() - template.cols() + 1, CvType.CV_32FC1);

        Imgproc.matchTemplate(image, template, matchResult, Imgproc.TM_CCOEFF);
        //Core.normalize(matchResult, matchResult, 0, 1, Core.NORM_MINMAX, -1, new Mat());

        Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(matchResult);
        Log.i("MainActivity", new Double(minMaxResult.maxVal).toString());
        if (minMaxResult.maxVal > 2E7) {
            return minMaxResult;
        } else {
            return null;
        }
    }


    @Override
    public boolean onTouch(View view, MotionEvent event) {
        maxValue = 0.0;
        return false;
    }
}
