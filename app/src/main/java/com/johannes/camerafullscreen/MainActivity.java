package com.johannes.camerafullscreen;

import android.graphics.Bitmap;
import android.media.Image;
import android.os.AsyncTask;
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
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, View.OnTouchListener {

    private CameraView cameraView;
    private ImageView resultView;
    private TextView curValueText;
    private TextView maxValueText;
    private Mat template = null;
    private Mat templateMask = null;
    private Mat sourceMask = null;
    private Mat sourceWatermark = null;
    private Mat matGray = null;
    private int matchFunction;

    private TemplateMatcherTask matcherTask;

    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == BaseLoaderCallback.SUCCESS) {
                cameraView.enableView();
                cameraView.setOnTouchListener(MainActivity.this);
                try {
                    template = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_template);
                    template = template.t();
                    templateMask = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_template_mask);
                    sourceMask = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_source_mask);
                    sourceWatermark = Utils.loadResource(mAppContext, R.drawable.marque_mercedes_source_watermark);
                    Imgproc.cvtColor(sourceWatermark, sourceWatermark, Imgproc.COLOR_GRAY2RGBA);
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

        resultView = (ImageView)findViewById(R.id.result_view);
        curValueText = (TextView)findViewById(R.id.curValue);
        maxValueText = (TextView)findViewById(R.id.maxValue);

        matchFunction = Imgproc.TM_CCORR_NORMED;
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

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (matcherTask == null || matcherTask.getStatus() == AsyncTask.Status.FINISHED) {
                    if (matcherTask != null) {
                        // process result
                        maxValueText.setText(TemplateMatcherTask.maxResult != null ? TemplateMatcherTask.maxResult.toString() : "");

                        // convert result image to bitmap
                        Mat resultImage = matcherTask.getResultImage();
                        Bitmap bmp = Bitmap.createBitmap(resultImage.cols(), resultImage.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(resultImage, bmp);
                        resultView.setImageBitmap(bmp);
                    }

                    matcherTask = new TemplateMatcherTask(template, templateMask, sourceMask, matchFunction, curValueText);
                    matcherTask.execute(matGray);
                }
            }
        });

        Mat matRgba = inputFrame.rgba();
        Mat matMasked = new Mat(matRgba.cols(), matRgba.rows(), CvType.CV_8UC1);
        matRgba.copyTo(matMasked, sourceMask);



        Core.addWeighted(matMasked, 1.0, sourceWatermark, 0.3, 0, matMasked);

        return matMasked;
    }


    @Override
    public boolean onTouch(View view, MotionEvent event) {
        TemplateMatcherTask.maxResult = null;
        return false;
    }
}
