package com.example.fakecheck;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.gun0912.tedpermission.PermissionListener;
import com.gun0912.tedpermission.TedPermission;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Vector;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

public class ViewActivity extends AppCompatActivity {
    MTCNN mtcnn;
    ImageView imageView;
    private static final String TAG = "fakecheck";
    String mCurrentPhotoPath;
    String img;
    Bitmap bitmapCrop;

    private Boolean isPermission = true;

    private static final int PICK_FROM_ALBUM = 1;
    private static final int PICK_FROM_CAMERA = 2;

    private File tempFile;
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.image_view);

        Button button = (Button)findViewById(R.id.button);
        tedPermission();

        Intent intent = getIntent();
        img = intent.getStringExtra("img");

        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                ByteArrayOutputStream stream= new ByteArrayOutputStream();
                Bitmap bit = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                float scale = (float)(1024/(float)bit.getWidth());
                int image_w = (int)(bit.getWidth()*scale);
                int image_h = (int)(bit.getHeight()*scale);
                Bitmap resize = Bitmap.createScaledBitmap(bit, image_w, image_h, true);
                resize.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray= stream.toByteArray();

                Intent intent = new Intent(ViewActivity.this, DeepActivity.class);
                intent.putExtra("image", byteArray);
                startActivity(intent);
            }
        });

        if(img.equals("cam")){//Camera버튼 클릭 시
            if(isPermission) takePhoto();
            else Toast.makeText(this.getApplicationContext(), getResources().getString(R.string.permission_2), Toast.LENGTH_LONG).show();
        }
        else if(img.equals("gal")){//Gallery버튼 클릭 시
            if(isPermission)  goToAlbum();
            else Toast.makeText(this.getApplicationContext(), getResources().getString(R.string.permission_2), Toast.LENGTH_LONG).show();
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != Activity.RESULT_OK) {
            Toast.makeText(this, "취소 되었습니다.", Toast.LENGTH_SHORT).show();

            if (tempFile != null) {
                if (tempFile.exists()) {
                    if (tempFile.delete()) {
                        Log.e(TAG, tempFile.getAbsolutePath() + " 삭제 성공");
                        tempFile = null;
                    }
                }
            }

            return;
        }

        if (requestCode == PICK_FROM_ALBUM) {

            Uri photoUri = data.getData();
            Log.d(TAG, "PICK_FROM_ALBUM photoUri : " + photoUri);

            Cursor cursor = null;

            try {

                /*
                 *  Uri 스키마를
                 *  content:/// 에서 file:/// 로  변경한다.
                 */
                String[] proj = {MediaStore.Images.Media.DATA};

                assert photoUri != null;
                cursor = getContentResolver().query(photoUri, proj, null, null, null);

                assert cursor != null;
                int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);

                cursor.moveToFirst();

                tempFile = new File(cursor.getString(column_index));
                Uri uri = FileProvider.getUriForFile(getBaseContext(), "com.example.fakecheck.fileprovider", tempFile);
                Log.d(TAG, "tempFile Uri : " + uri);

            } finally {
                if (cursor != null) {
                    cursor.close();
                }
            }

            setImage();

        } else if (requestCode == PICK_FROM_CAMERA) {
            setImage();

        }
    }


    //앨범에서 이미지 가져오기
    private void goToAlbum() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        startActivityForResult(intent, PICK_FROM_ALBUM);
    }


    //카메라로 이미지 가져오기
    private void takePhoto() {

        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        try {
            tempFile = createImageFile();
        } catch (IOException e) {
            Toast.makeText(this, "이미지 처리 오류! 다시 시도해주세요.", Toast.LENGTH_SHORT).show();
            finish();
            e.printStackTrace();
        }
        if (tempFile != null) {
            Uri photoUri = FileProvider.getUriForFile(getBaseContext(), "com.example.fakecheck.fileprovider", tempFile);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            startActivityForResult(intent, PICK_FROM_CAMERA);
        }
    }

    //폴더 및 파일 만들기
    private File createImageFile() throws IOException {

        // 이미지 파일 이름 ( fakecheck_{시간}_ )
        String timeStamp = new SimpleDateFormat("HHmmss").format(new Date());
        String imageFileName = "fakecheck_" + timeStamp + "_";

        // 이미지가 저장될 폴더 이름 ( fakecheck )
        File storageDir = new File(Environment.getExternalStorageDirectory() + "/fakecheck/");
        if (!storageDir.exists()) storageDir.mkdirs();

        // 파일 생성
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);
        mCurrentPhotoPath = image.getAbsolutePath();
        Log.d(TAG, "createImageFile : " + image.getAbsolutePath());

        return image;
    }

    //tempFile->bitmap->crop image를 ImageView 에 설정
    private void setImage() {
        imageView=(ImageView)findViewById(R.id.imageView);
        BitmapFactory.Options options = new BitmapFactory.Options();
        Bitmap originalBm = BitmapFactory.decodeFile(tempFile.getAbsolutePath(), options);
        Log.d(TAG, "setImage : " + tempFile.getAbsolutePath());
        mtcnn=new MTCNN(getAssets());
        if(img.equals("cam")) {//카메라로 받아오는 경우 이미지가 회전해서 들어옴
            try {
                ExifInterface ei = new ExifInterface(mCurrentPhotoPath);
                int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);
                Bitmap rotatedBitmap = null;
                switch (orientation) {
                    //이미지가 회전된 만큼 반대로 회전
                    case ExifInterface.ORIENTATION_ROTATE_90:
                        rotatedBitmap = rotateImage(originalBm, 90);
                        break;

                    case ExifInterface.ORIENTATION_ROTATE_180:
                        rotatedBitmap = rotateImage(originalBm, 180);
                        break;

                    case ExifInterface.ORIENTATION_ROTATE_270:
                        rotatedBitmap = rotateImage(originalBm, 270);
                        break;

                    case ExifInterface.ORIENTATION_NORMAL:
                    default:
                        rotatedBitmap = originalBm;
                }
                originalBm = rotatedBitmap;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        processImage(originalBm);
        /*tempFile 사용 후 null 처리 필요
         *(resultCode != RESULT_OK) 일 때 tempFile 을 삭제하기 때문에
         *기존에 데이터가 남아 있게 되면 원치 않는 삭제가 이루어짐
         */
        tempFile = null;

    }

    //권한 설정
    private void tedPermission() {

        PermissionListener permissionListener = new PermissionListener() {
            @Override
            public void onPermissionGranted() {
                // 권한 요청 성공
                isPermission = true;

            }

            @Override
            public void onPermissionDenied(ArrayList<String> deniedPermissions) {
                // 권한 요청 실패
                isPermission = false;

            }
        };

        TedPermission.with(this)
                .setPermissionListener(permissionListener)
                .setRationaleMessage("카메라 권한이 필요합니다.")
                .setDeniedMessage(getResources().getString(R.string.permission_1))
                .setPermissions(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA)
                .check();

    }

    //MTCNN
    public void processImage(Bitmap bitmap){
        Bitmap bm= Utils.copyBitmap(bitmap);
        Vector<Box> boxes = mtcnn.detectFaces(bm, bm.getWidth()/5);
        if(boxes.toString()=="[]"){//인식된 얼굴이 없는 경우
            Intent intent = new Intent(getApplicationContext(), MainActivity.class);
            startActivity(intent);
            Toast.makeText(this.getApplicationContext(), "얼굴이 아닙니다.", Toast.LENGTH_LONG).show();
        }
        else{//얼굴이 인식된 경우
            Box box = boxes.get(0);
            bm = face_align(bm, box.landmark);
            boxes = mtcnn.detectFaces(bm, bm.getWidth()/5);
            box = boxes.get(0);
            box.toSquareShape();
            box.limitSquare(bm.getWidth(), bm.getHeight());
            Rect rect = box.transform2Rect();
            bitmapCrop = crop(bm, rect);
            imageView.setImageBitmap(bitmapCrop);
        }
    }
    /*
    이미지 회전
     */
    public static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }
    /*
    인식된 얼굴 크기만큼 crop
     */
    public static Bitmap crop(Bitmap bitmap, Rect rect){
        Bitmap croped = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top);
        return croped;
    }
    /*
    얼굴 인식 후 정면으로 회전
     */
    public static Bitmap face_align(Bitmap bitmap, Point[] landmarks) {
        float diffEyeX = landmarks[1].x - landmarks[0].x;
        float diffEyeY = landmarks[1].y - landmarks[0].y;

        float fAngle;
        if (Math.abs(diffEyeY) < 1e-7) {
            fAngle = 0.f;
        } else {
            fAngle = (float) (Math.atan(diffEyeY / diffEyeX) * 180.0f / Math.PI);
        }
        Matrix matrix = new Matrix();
        matrix.setRotate(-fAngle);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
}
