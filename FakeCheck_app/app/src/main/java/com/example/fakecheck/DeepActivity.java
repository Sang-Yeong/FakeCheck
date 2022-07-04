package com.example.fakecheck;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import androidx.appcompat.app.AppCompatActivity;

public class DeepActivity extends AppCompatActivity {
    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.deepfake_view);

        ImageView imageview = (ImageView) findViewById(R.id.imageView2);
        textView = (TextView) findViewById(R.id.textView2);

        byte[] byteArray = getIntent().getByteArrayExtra("image");
        Bitmap bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        int[] pixels = new int[224*224];
        bitmap.getPixels(pixels,0,224,0,0,224,224);
        int k=0;
        //이미지 input은 BGR로 들어가야함.
        float[][][][] input = new float[1][3][224][224];
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int pixel = pixels[k++];
                input[0][2][y][x]=((pixel>>0)&0xff)/(float)255;
                input[0][1][y][x]=((pixel>>8)&0xff)/(float)255;
                input[0][0][y][x]=((pixel>>16)&0xff)/(float)255;
            }
        }
        Interpreter tf_lite = getTfliteInterpreter("tfLite_model2.tflite");
        float[][] output = new float[1][2];
        tf_lite.run(input, output);
        //결과 출력
        if(output[0][0]<output[0][1]){
            textView.setText("real");
        }
        else if(output[0][0]>output[0][1]){
            textView.setText("fake");
        }

    }
    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(DeepActivity.this, modelPath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}