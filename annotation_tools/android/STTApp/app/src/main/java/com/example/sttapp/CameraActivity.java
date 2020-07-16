/*
 * CameraActivity manages the camera functionality and detects QR codes. If there is a valid QR code with the information, this activity closes and sends the info (address/port) to the MainActivity.
 */

package com.example.sttapp;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcode;
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcodeDetector;
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcodeDetectorOptions;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


public class CameraActivity extends AppCompatActivity {


    private static final String TAG = "CameraActivity";

    protected CameraDevice camera;
    private Handler backgroundHandler;
    private HandlerThread backgroundThread;
    private Size imageDimension;
    protected CaptureRequest.Builder captureRequestBuilder;
    protected CaptureRequest captureRequest;
    private static final int CAMERA_PERMISSION = 200;
    protected CameraCaptureSession cameraCaptureSession;
    private String cameraId;
    private TextureView textureView;
    private ImageReader imageReader;


    static class CompareSizesByArea implements Comparator<Size> {
        /*
         * compare 2 Sizes based on their areas.
         */
        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        /*
         * After user has not granted permission to use camera, go back to MainActivity and send back empty fields
         */
        if (requestCode == CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                //go back
                Log.d(TAG,"camera permission denied");
                String[] empty = {};
                backToSettingsActivity(empty);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        /*
         * connect the textureView from the CameraActivity.xml and set the listener
         */
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        textureView = (TextureView) findViewById(R.id.textureview_id);
        textureView.setSurfaceTextureListener(textureListener);
    }


    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int width, int height) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i, int i1) { }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) { return true; }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) { }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            camera = cameraDevice;
            createCameraPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            camera.close();
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int i) {
            camera.close();
            camera = null;
        }
    };

    protected void startBackgroundThread() {
        /*
         * start a background thread for the camera
         */
        backgroundThread = new HandlerThread("Camera Background");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    protected void stopBackgroundThread() {
        /*
         * quits the background thread that is currently running
         */
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected void createCameraPreview() {
        /*
         * set up captureRequests to send each frame of camera input to the surface for live preview and to the imageReader to be processed (detect QRCode)
         */
        try {
            SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(surfaceTexture);
            captureRequestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface); // show the camera output (preview) on screen
            captureRequestBuilder.addTarget(imageReader.getSurface()); // process each frame (check if qrcode)
            camera.createCaptureSession(Arrays.asList(surface, imageReader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession ccs) {
                    if (null == camera) { // camera is closed already
                        return;
                    }
                    cameraCaptureSession = ccs;
                    try {
                        captureRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                        captureRequest = captureRequestBuilder.build();
                        cameraCaptureSession.setRepeatingRequest(captureRequest, null, backgroundHandler);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Log.d(TAG,"onConfigureFailed in createCameraPreview");
                }
            }, backgroundHandler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void openCamera() {
        /*
         * set up and open the camera- figure out height and width of image and check that permissions are granted
         * initialize the imageReader to obtain each frame image and process to try to detect QR Code
         */
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraId = cameraManager.getCameraIdList()[0];
            CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId);

            StreamConfigurationMap map = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];

            textureView.setMinimumWidth(imageDimension.getWidth());
            textureView.setMinimumHeight(imageDimension.getHeight());

            //camera permissions
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                //permission not granted
                ActivityCompat.requestPermissions(CameraActivity.this,new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
                return;
            }

            Size largest = Collections.max(Arrays.asList(map.getOutputSizes(ImageFormat.YUV_420_888)), new CompareSizesByArea());

            imageReader = ImageReader.newInstance(largest.getWidth() / 4, largest.getHeight() / 4, ImageFormat.YUV_420_888, 2);

            ImageReader.OnImageAvailableListener imageAvailableListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader imageReader) {
                    Image image = imageReader.acquireNextImage();
                    if (image == null) {
                        return;
                    } else {
                        detectQRcode(image);
                        image.close();
                    }
                }
            };
            imageReader.setOnImageAvailableListener(imageAvailableListener, backgroundHandler);
            cameraManager.openCamera(cameraId,stateCallback,backgroundHandler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        /*
         * closes the camera if one is open
         */
        if (null != camera) {
            camera.close();
            camera = null;
        }
    }

    @Override
    protected void onResume() {
        /*
         * camera2 onResume function, make sure the live preview continues
         */
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        /*
         * camera2 onPause function, close the camera / stop running background thread when paused
         */
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void detectQRcode(Image image) {
        /*
         * given an image, using Firebase ML Kit, detect if there is a valid QRCode and extract the information
         * if the information is received successfully, go back to MainActivity and send it the info
         */
        FirebaseVisionBarcodeDetectorOptions options = new FirebaseVisionBarcodeDetectorOptions.Builder().setBarcodeFormats(FirebaseVisionBarcode.FORMAT_QR_CODE).build();
        FirebaseVisionImage firebaseimage = FirebaseVisionImage.fromMediaImage(image, 0);

        FirebaseVisionBarcodeDetector detector = FirebaseVision.getInstance().getVisionBarcodeDetector(options);

        Task<List<FirebaseVisionBarcode>> result = detector.detectInImage(firebaseimage)
                .addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionBarcode>>() {
                    @Override
                    public void onSuccess(List<FirebaseVisionBarcode> barcodes) {
                        if (barcodes.size() < 1) {
                            Log.d(TAG,"no qrcode detected");
                        } else {
                            // Task completed successfully
                            for (FirebaseVisionBarcode barcode : barcodes) {
                                // can be used for a graphical overlay (not implemented)
                                Rect bounds = barcode.getBoundingBox();
                                Point[] corners = barcode.getCornerPoints();

                                String rawValue = barcode.getRawValue();
                                Log.d(TAG, "qrcode detected: " + rawValue);

                                String[] info = rawValue.split(",");
                                if (info.length <= 1) {
                                    Log.d(TAG,"valid qr code but info not formatted correctly");
                                } else {
                                    backToSettingsActivity(info);
                                }
                            }
                        }
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        // Task failed with an exception
                        Log.d(TAG, "qrcode detector failure");
                    }
                });

    }

    private void backToSettingsActivity(String[] info) {
        /*
         * create a intent to go back to the settings activity and pass the address and port info
         */
        Intent intent = new Intent();
        String ipaddress = "";
        String port = "";
        String username = "";

        if (info.length > 0) {
            ipaddress = info[0];
            if (info.length > 1) {
                port = info[1];
                if (info.length > 2) {
                    username = info[2];
                }
            }
        }
        intent.putExtra("ipaddress",ipaddress);
        intent.putExtra("port",port);
        intent.putExtra("username", username);
        setResult(RESULT_OK,intent);
        finish();
    }

}

